# HA-pok：PTCG 强化学习 V1–V3

面向 Kaggle PTCG AI Battle Challenge Playground 的候选动作评分式 Actor–Critic / PPO 实验代码。三个目录均含独立源码和使用说明。

|版本|主要变化|已完成的历史正式训练|
|---|---|---|
|[V1](v1/README.md)|基础编码，主要学习强制单选|10,000局|
|[V2](v2/README.md)|无放回顺序选择、STOP、卡牌实体解析|10,000局|
|[V3](v3/README.md)|己方弃牌身份、能量统计、攻击相关特征|20,000局|

所有正式训练均为固定官方样例卡组、双方共享策略的BO1自博弈。没有历史策略池、搜索树或BO3训练；代码可运行不代表竞技能力。线上积分随匹配变化，因此不把历史分数当作固定benchmark。

## 1. 任务、数据与系统结构

本项目不从卡牌CSV直接学习动作，也不使用LLM生成的动作进行PPO更新。训练数据由官方本地引擎实时生成：

```text
官方样例卡组 × 双方共享策略
             ↓
引擎返回行动方观测与合法候选
             ↓
状态/候选编码 → Actor采样 → 引擎执行
             ↓
完整对局终局回报 → PPO更新 → 下一批新对局
             ↓
checkpoint → NumPy导出 → 本地验证 → 手动提交Kaggle
```

卡牌元数据用于V3特征编码；下载的线上回放用于诊断和回归分析，没有直接作为PPO样本。仓库不包含这些数据。V1–V3的历史实验均从各版本新参数开始训练，不是逐版本自动续训。

## 2. 数学建模：部分可观测双人博弈

设引擎完整状态为 $x_t$，当前行动方为 $p_t\in\{0,1\}$。玩家只能获得自身可见观测：

$$
o_t=\Omega(x_t,p_t),\qquad x_{t+1}\sim P(\cdot\mid x_t,A_t).
$$

$A_t$ 是提交给引擎的候选索引列表，而不一定是单个动作。完整问题具有隐藏信息；目前用前馈网络近似无记忆策略，不显式维护对手手牌的信念分布：

$$
s_t=f(o_t),\qquad \pi_\theta(A_t\mid o_t).
$$

这里 $s_t$ 是编码后的观测，不应误称为环境完整Markov状态。相同编码可能对应不同隐藏状态，甚至因编码缺失对应不同可见局面。

为了解释优化目标，先固定对手策略 $\mu$：

$$
J(\theta;\mu)=\mathbb E_{\tau\sim(\pi_\theta,\mu,P)}[R^{(p)}(\tau)].
$$

实际训练的双方共享当前策略，采样一批期间权重冻结，更新后对手也随之变化。因此这是同时自博弈的实践近似，不是求解固定对手的稳定优化，也没有纳什均衡或单调胜率提升保证。不能把双方零和回报相加后当作一个待最大化的联合收益——该和恒为零。

## 3. 特征设计与版本演化

|版本|状态维度|基础候选维度|网络候选输入维度|核心区别|
|---|---:|---:|---:|---|
|V1|128|80|80|压缩特征，主要训练强制单选|
|V2|2392|208|416|实体解析，顺序多选与STOP|
|V3|3364|228|456|己方弃牌身份、基本能量及攻击条件|

### 3.1 V2卡牌与状态编码

卡牌ID使用16位二进制表示，低位在前：

$$
b_j(u)=(u\gg j)\mathbin{\&}1,\qquad j=0,\ldots,15.
$$

在支持的 $0\le u<65536$ 范围内避免ID取模冲突，但二进制身份不等于卡牌效果语义。实体编码为：

$$
\phi(c)=\left[b_{16}(id),\frac{hp}{400},\frac{maxHp}{400},
\mathbf1_{\mathrm{appearThisTurn}},\frac{n_{tools}}4,
\frac{n_{energy,0}}8,\ldots,\frac{n_{energy,11}}8\right]\in\mathbb R^{32}.
$$

状态包含24维全局统计、64维选择上下文、双方各6个场面槽位、己方手牌与对方弃牌的排序ID序列：

$$
d_s^{(2)}=24+64+12\times32+60\times16+60\times16=2392.
$$

场面每方仅完整编码1个战斗位和前5个备战位。分母用于缩放，不做数值裁剪。缺失实体补零，某些类别越界会夹到边界；并非完整无损观测。

基础候选编码包含动作类别、选择上下文、来源/目标/效果实体、攻击ID及数值字段：

$$
d_q^{(2)}=18+64+3\times32+16+14=208.
$$

其中区域和索引先解析到卡牌实体，避免把手牌槽位误当成卡牌身份。对手隐藏手牌不编码。

### 3.2 V3为何需要增加弃牌特征

Riptide的伤害分量依赖己方弃牌堆基本水能量数量 $n_W$：

$$
D_{\mathrm{Riptide,component}}=20n_W.
$$

V2只记录己方弃牌数量，无法区分“弃掉一张训练家”和“弃掉一张水能量”。若两种局面编码相同，前馈网络不可能据此给出不同预测；仅增加训练局数无法消除这种信息丢失。

V3追加己方弃牌ID序列与12类基本能量计数：

$$
d_s^{(3)}=2392+60\times16+12=3364.
$$

每个候选追加20维：攻击元数据存在标志、基础伤害、12类能量费用、Riptide公式已知标志及伤害分量、Hammer-lanche标志/将弃张数/低牌库标志、效果文本存在标志。新增字段置于四个选择进度槽位之前。

这些字段不是完整伤害模拟器：不统一计算弱点、抗性、工具、防御或随机牌库顶。Hammer-lanche仅标注弃顶六张与牌库不足风险，不偷看牌库顺序。引擎元数据随checkpoint保存，部署时嵌入导出源码。

## 4. 组合动作：顺序无放回选择

引擎给出 $N$ 个候选 $C=\{0,\ldots,N-1\}$，需要选择 $m$ 至 $M$ 项。若一次直接枚举无序组合，数量为：

$$
\sum_{j=m}^{M}\binom Nj.
$$

V2/V3不枚举所有集合，而将一次选择分解成若干微决策。已选前缀为 $S_k$、数量为 $k$：

$$
\mathcal U_k=(C\setminus S_k)\cup
\begin{cases}
\{\mathrm{STOP}\},&k\ge m,\\
\varnothing,&k<m.
\end{cases}
$$

选中候选后移除它，防止重复；选STOP结束，达到 $M$ 自动结束。$M=0$ 时不产生微决策。满足最小数量之前STOP不进入动作集，合法性通过候选构造保证，而不是依赖网络自行学习规则。

### 4.1 前缀条件化

基础候选向量为 $q(o,a)$。已选集合以均值表示：

$$
\bar q(S_k)=\begin{cases}
\frac1k\sum_{a\in S_k}q(o,a),&k>0,\\
\mathbf0,&k=0.
\end{cases}
\qquad \tilde q_{k,a}=[q_k(o,a);\bar q(S_k)].
$$

当前候选还写入已选数量、剩余选择上下限和候选数量。状态中的原选择上下限改为剩余数量。因此网络候选输入为V2的416维或V3的456维。

前缀均值不保留顺序，且不同集合可能有相同均值，这是表达能力限制。每次仍重新评分剩余候选，计算量约与 $\sum_k|\mathcal U_k|$ 成正比，而不是遍历所有组合。

### 4.2 路径概率与集合概率

实际顺序路径 $z=(z_0,\ldots,z_{K-1})$ 的概率为：

$$
P_\theta(z\mid o)=\prod_{k=0}^{K-1}\pi_\theta(z_k\mid s_k,Q_k),
\qquad
\log P_\theta(z\mid o)=\sum_k\log\pi_\theta(z_k\mid s_k,Q_k).
$$

无序集合的概率还要对产生该集合的全部路径求和。实现没有计算这个和，而是把每个实际微决策当作PPO样本。微步骤上的裁剪目标不等价于对整个组合动作概率比只裁剪一次。

## 5. 候选评分式 Actor–Critic

网络隐藏宽度128，激活Tanh；V2共425,858参数。状态与候选分别编码：

$$
h_s=\tanh(W_{s2}\tanh(W_{s1}s+b_{s1})+b_{s2}),
\qquad h_a=\tanh(W_a\tilde q_a+b_a).
$$

策略头为每个合法候选输出一个logit：

$$
\ell_a=w_\ell^\top\tanh(W_\ell[h_s;h_a]+b_\ell)+b_{out},
\qquad
\pi_\theta(a\mid s,Q)=\frac{e^{\ell_a}}{\sum_{b\in\mathcal U}e^{\ell_b}}.
$$

变长候选在训练batch中补齐，padding logit置为浮点最小有限值，使其概率近似为零。候选共享评分网络，没有候选位置embedding。

价值头预测当前行动方的终局回报：

$$
V_\theta(s)=w_v^\top\tanh(W_vh_s+b_v)+b_{v,out}.
$$

当前价值头只读状态，不读候选或已选卡牌前缀；不同前缀可能获得相同价值预测。训练按categorical采样，部署逐步取argmax；部署策略与采样策略的实际强度需分别评估。

## 6. 从策略梯度到本项目的 PPO

以下终局/截断处理主要对应V2/V3。V1历史采集器对非0/1终局未专门赋平局0回报，达到步数上限抛错；不要认为三个版本所有细节完全一致。

### 6.1 Monte Carlo回报

对玩家 $p$ 的微决策 $i$，使用所属完整对局的结果：

$$
G_i^{(p)}=\begin{cases}
+1,&\text{玩家 }p\text{ 获胜},\\
-1,&\text{对方获胜},\\
0,&\text{平局}.
\end{cases}
$$

没有中间奖励、折扣或bootstrap，等价于终局奖励且 $\gamma=1$。V2/V3超过2000次引擎选择仍未终局的轨迹整局丢弃，不当作负样本。

### 6.2 似然比梯度与基线

对固定对手目标，利用 $\nabla P=P\nabla\log P$ 可得策略梯度的采样形式：

$$
\nabla_\theta J\approx
\mathbb E\left[\sum_{i\in\mathcal I_p}
\nabla_\theta\log\pi_\theta(a_i\mid s_i,Q_i)G_i\right].
$$

$\mathcal I_p$ 为该玩家实际产生的微决策。引入与采样动作无关的价值基线以降低方差：

$$
\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s,Q)V(s)]
=V(s)\nabla\sum_a\pi(a\mid s,Q)=0.
$$

因此使用行为策略采样时保存的价值估计计算优势：

$$
A_i=G_i-V_{\theta_{old}}(s_i).
$$

实现进一步按每个minibatch标准化，而不是全轨迹统一标准化：

$$
\hat A_i=\frac{A_i-\operatorname{mean}_B(A)}{\operatorname{std}_B(A)+10^{-8}}.
$$

标准差采用总体标准差。该标准化是优化启发式；不将其表述为保持有限样本梯度严格不变。

### 6.3 旧策略概率与裁剪

在旧策略数据上计算新旧动作概率比：

$$
r_i(\theta)=\exp\left[
\log\pi_\theta(a_i\mid s_i,Q_i)-\log\pi_{\theta_{old}}(a_i\mid s_i,Q_i)
\right].
$$

直接最大化 $\mathbb E[r_i\hat A_i]$ 容易在重复利用样本时产生过大的策略更新。PPO使用裁剪代理目标：

$$
L^{clip}=\mathbb E_B\left[
\min\left(r_i\hat A_i,
\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)\hat A_i\right)
\right],\qquad\epsilon=0.2.
$$

当优势为正，超过 $1+\epsilon$ 后不继续奖励增大该动作概率；当优势为负，低于 $1-\epsilon$ 后不继续奖励减小概率。它是代理目标的限制，不是对真实KL散度或胜率的硬保证。

价值损失与策略熵分别为：

$$
L_V=\mathbb E_B[(V_\theta(s_i)-G_i)^2],\qquad
H=\mathbb E_B\left[-\sum_a\pi_\theta(a)\log\pi_\theta(a)\right].
$$

代码通过梯度下降最小化：

$$
\boxed{L=-L^{clip}+0.5L_V-0.01H}.
$$

实现位置：各版本的`src/ptcg_rl/ppo.py`。没有GAE、学习率调度、价值裁剪或KL early stopping。

### 6.4 在线采样为何重要

V2/V3在整批采样期间保持权重不变，记录实际采样动作的log-probability。buffer达到至少1024条微决策后进行2个epoch更新，每batch最多128条，随后清空buffer，重新采样。更新前抽查前10条行为概率差小于 $10^{-4}$。

Adam学习率为 $3\times10^{-4}$，梯度全局范数裁剪为1。minibatch按采样顺序遍历，没有shuffle；每个微决策等权，较长多选过程贡献更多样本。

线上日志中的贪心动作不能直接伪装成随机行为策略样本。用当前网络补算一个概率并不能恢复原来的采样分布；这也是本项目没有将Kaggle回放直接送入PPO的原因。

## 7. 训练记录和评估边界

|版本|正式对局数|微决策样本数|optimizer step|
|---|---:|---:|---:|
|V1|10,000|历史统计口径与V2不同，不作直接比较|不在此汇总|
|V2|10,000|404,937|6,928|
|V3|20,000|887,557|15,140|

这些是本地历史运行记录，不是性能对照实验。V2/V3改变特征且训练局数不同，不能单凭线上分数变化归因于编码改进。同一策略自博弈的座位胜负统计也不是对外胜率。

要检验提升，应冻结测试对手、交换座位、报告对局数量，并使用独立新对局。当前尚未实现这个完整评估框架。静态回放只适合接口/特征回归；换动作后的轨迹不能直接沿用原回放推断胜负。

## 8. 安装和比赛资源

推荐Python 3.11。在仓库根目录执行 `python -m pip install -r requirements.txt`。依赖范围不是精确锁定环境；PyTorch请按自身CPU/GPU平台选择安装方式。现有训练使用CPU单线程。

自行接受比赛条款并下载官方资源，设其解压根目录为 `$data`。目录应包含 `sample_submission/sample_submission/sample_submission/cg` 及同级 `deck.csv`。各版本命令显式传入 `--competition-root`，无需复用作者电脑路径。

仓库不包含官方引擎二进制、卡牌数据库、下载回放、认证文件、模型权重或生成提交包；这些文件的获取和使用受各自条款约束。V3导出时会把本地引擎元数据嵌入自己的提交包，因此生成包也不纳入Git。

### V3 快速开始

从仓库根目录执行安装后：

```powershell
cd v3
$data = 'D:\your-data\the-pokemon-company-ptcg-ai-battle-challenge-playground'
# 先做少量烟雾训练，确认引擎和依赖可用
python scripts/train_v2.py --competition-root $data --feature-version v3 --games 2 --output artifacts/smoke.pt
python scripts/export_v2.py --checkpoint artifacts/smoke.pt
python scripts/verify_submission.py --competition-root $data --submission-dir submission_v3 --archive submission_v3/submission.tar.gz
# 确认无误后正式训练；需重新导出正式checkpoint，不能提交smoke包
python scripts/train_v2.py --competition-root $data --feature-version v3 --games 20000 --output artifacts/policy_v3_20k.pt
```

V2/V3保留共享脚本名`train_v2.py`和`export_v2.py`，通过feature-version和checkpoint schema确定版本。更详细命令见各版本README。导出包含40组确定性随机输入的PyTorch/NumPy一致性检查，完整游戏合法性另由verify_submission验证。

## 9. 仓库组织与公式对应

```text
v1/  基础编码、单选PPO、训练与部署模板
v2/  顺序多选、STOP与实体特征
v3/  弃牌与攻击条件特征
  src/ptcg_rl/v3.py       f(o)、q(o,a) 与顺序候选集合
  src/ptcg_rl/v2.py       基础编码与共享选择循环
  src/ptcg_rl/policy.py   πθ 和 Vθ
  src/ptcg_rl/ppo.py      Lclip、LV、H 和优化器更新
  scripts/train_v2.py     在线轨迹、Gi、Ai、checkpoint
  scripts/export_v2.py    NumPy推理与schema校验
  scripts/verify_submission.py  提交包和完整局验证
```

V2/V3目录包含共享兼容模块；不要仅根据旧文件名判断实际算法版本。

## 10. 安全与复现边界

- Kaggle认证在本机完成，不把token写入源码或提交到Git。
- 只加载可信来源的PyTorch checkpoint；当前加载使用`weights_only=False`。
- 各版本权重不直接兼容。训练默认从随机参数开始，非自动续训。
- checkpoint仅在训练结束保存；没有自动恢复或中途checkpoint。
- NumPy与PyTorch种子受控，官方引擎随机性未显式设种，因此不是逐局确定性复现。
- 本仓库是当前工作代码按版本分组的整理版，不是历史Git提交的逐字快照。包括后续修正的统计和加载逻辑。
- 已停止的DeepSeek/LLM实验不在本次上传范围。

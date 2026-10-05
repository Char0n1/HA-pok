# HA-pok：PTCG 强化学习 V1–V3

面向 Kaggle PTCG AI Battle Challenge Playground 的候选动作评分式 Actor–Critic / PPO 实验代码。三个目录均含独立源码和使用说明。

|版本|主要变化|已完成的历史正式训练|
|---|---|---|
|[V1](v1/README.md)|基础编码，主要学习强制单选|10,000局|
|[V2](v2/README.md)|无放回顺序选择、STOP、卡牌实体解析|10,000局|
|[V3](v3/README.md)|己方弃牌身份、能量统计、攻击相关特征|20,000局|

所有正式训练均为固定官方样例卡组、双方共享策略的BO1自博弈。没有历史策略池、搜索树或BO3训练；代码可运行不代表竞技能力。线上积分随匹配变化，因此不把历史分数当作固定benchmark。

## 安装和比赛资源

推荐Python 3.11。在仓库根目录执行 `python -m pip install -r requirements.txt`。依赖范围不是精确锁定环境；PyTorch请按自身CPU/GPU平台选择安装方式。现有训练使用CPU单线程。

自行接受比赛条款并下载官方资源，设其解压根目录为 `$data`。目录应包含 `sample_submission/sample_submission/sample_submission/cg` 及同级 `deck.csv`。各版本命令显式传入 `--competition-root`，无需复用作者电脑路径。

仓库不包含官方引擎二进制、卡牌数据库、下载回放、认证文件、模型权重或生成提交包；这些文件的获取和使用受各自条款约束。V3导出时会把本地引擎元数据嵌入自己的提交包，因此生成包也不纳入Git。

## 安全与复现边界

- Kaggle认证在本机完成，不把token写入源码或提交到Git。
- 只加载可信来源的PyTorch checkpoint；当前加载使用`weights_only=False`。
- 各版本权重不直接兼容。训练默认从随机参数开始，非自动续训。
- checkpoint仅在训练结束保存；没有自动恢复或中途checkpoint。
- NumPy与PyTorch种子受控，官方引擎随机性未显式设种，因此不是逐局确定性复现。
- 本仓库是当前工作代码按版本分组的整理版，不是历史Git提交的逐字快照。包括后续修正的统计和加载逻辑。
- 已停止的DeepSeek/LLM实验不在本次上传范围。

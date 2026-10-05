# V1：基础候选评分 PPO

## 方法与代码注释

`src/ptcg_rl/features.py`编码观测和候选，`policy.py`实现共享MLP Actor–Critic；`selfplay.py`采样对局，`ppo.py`实现clip PPO。策略只对引擎合法候选评分，不预测全局动作编号。

历史正式实验为10,000局。重要局限：`minCount=0`直接选空，多选通常只取最小数量，PPO主要学习恰好强制单选的决策；卡牌ID压缩存在混叠。保留用于基线比较，不推荐作为新训练主线。

## 使用

先在仓库根目录安装依赖，再进入本目录；`$data`替换为官方比赛资源解压根目录的绝对路径。

```powershell
cd v1
$data = 'D:\your-data\the-pokemon-company-ptcg-ai-battle-challenge-playground'
python scripts/smoke_test.py --competition-root $data
python scripts/stream_train.py --competition-root $data --games 10000
python scripts/build_submission.py --competition-root $data --checkpoint artifacts/policy_10k_selfplay.pt
python scripts/verify_submission.py --competition-root $data
```

训练脚本输出路径以其参数和终端提示为准；默认文件名可能不同，导出时将checkpoint替换为实际生成文件。`python scripts/stream_train.py --help`可查看完整参数。

`submission_template/main.py`是NumPy部署模板，构建产物在`submission/`。`submit_kaggle.py --submit`会实际上传，请只在认证及验证完成后主动执行。

## 离线采集路径

`collect_selfplay.py`保存行为模型权重和轨迹；`train_ppo.py`要求行为权重，不能将来自其他未知策略的动作概率配上随机新模型。早期不含行为权重的旧轨迹不兼容。优先使用在线`stream_train.py`。

## 版本边界

含后续修复：真实玩家索引胜负统计、raw-exec资源加载兼容、行为权重一致性检查。不是初版缺陷的逐字历史镜像。权重不随仓库发布，也不能加载到V2/V3。

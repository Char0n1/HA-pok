# V3：弃牌身份与攻击条件特征

在V2之上增加己方弃牌最多60个排序ID及12类基本能量计数。状态3364维；基础候选228维，拼接前缀后456维。schema为`ptcg-v3-discard-attacks-1`，不能直接加载V2权重。

## 新特征和实现位置

`src/ptcg_rl/v3.py`组合V2编码和新的字段，`configure()`加载官方引擎卡牌/攻击目录。20个新增候选字段包括：元数据存在标志、基础伤害、12类费用、Riptide的水能量伤害分量、Hammer-lanche弃牌库和低牌库风险、效果文本存在标志。

伤害字段是基础分量，不处理全部工具、弱点、抗性或防御效果，不是完整伤害模拟器。仅针对两种当前卡组核心效果有明确动态公式，未实现通用卡牌文本编码；不会读取隐藏牌库顶。

## 训练与部署

```powershell
cd v3
$data = 'D:\your-data\the-pokemon-company-ptcg-ai-battle-challenge-playground'
python scripts/train_v2.py --competition-root $data --feature-version v3 --games 20000 --output artifacts/policy_v3_20k.pt
python scripts/export_v2.py --checkpoint artifacts/policy_v3_20k.pt
python scripts/verify_submission.py --competition-root $data --submission-dir submission_v3 --archive submission_v3/submission.tar.gz
```

脚本保留共享入口名字；本目录训练默认feature-version=v3，仍建议显式指定。checkpoint保存引擎目录，导出时嵌入main.py，因此推理不用DLL；模型训练与本地完整对局验证仍需官方引擎。

历史正式训练：20,000局，887,557条微决策，15,140次参数更新，截断0局。正式权重未纳入仓库。训练仍从随机初始化开始，固定同卡组BO1自博弈；奖励和PPO参数沿用V2，没有增加对手池或奖励塑形。

## 验证与使用边界

仓库导出执行40组确定性随机输入的PyTorch/NumPy数值检查，不依赖外部回放。verify_submission检查压缩包与目录字节一致、无__file__加载、完整局动作合法性。推荐先用`--games 2`跑烟雾测试再正式训练；不要把烟雾模型直接上传。

V3保留V2基础模块及旧features/policy依赖以保证独立运行。所增加的特征解决已确认的信息缺失，但正式训练完成和成功上传都不是胜率提升的证明。

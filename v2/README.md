# V2：顺序无放回选择与 STOP

状态2392维；基础候选208维，与已选前缀均值拼接后416维。128宽度Tanh MLP Actor–Critic，共425,858参数。schema为`ptcg-v2-sequential-1`。

`v2.py`编码卡牌来源、目标、选择上下文；用16位ID替代取模。多选拆为微决策，满足minCount才允许STOP，达到maxCount自动结束。训练采样，部署贪心；所有微决策参与PPO。

## 训练与导出

```powershell
cd v2
$data = 'D:\your-data\the-pokemon-company-ptcg-ai-battle-challenge-playground'
python scripts/train_v2.py --competition-root $data --feature-version v2 --games 10000 --output artifacts/policy_v2_10k.pt
python scripts/export_v2.py --checkpoint artifacts/policy_v2_10k.pt
python scripts/verify_submission.py --competition-root $data --submission-dir submission_v2 --archive submission_v2/submission.tar.gz
```

历史正式训练：10,000局、404,937条微决策、6,928次optimizer step。该权重不包含在仓库。

导出使用同一编码/选择源码和NumPy权重。仓库版数值一致性测试使用40组确定性随机输入，不依赖作者私有回放；另用verify_submission在本地引擎完成一局并检查动作合法性。

## PPO 配置

Adam学习率3e-4，clip=0.2，value系数0.5，entropy系数0.01，梯度范数1。完整局终局奖励±1/0，gamma=1 Monte Carlo优势，无GAE。buffer至少1024微决策后更新，两epoch、batch128，未洗牌。超过2000引擎选择未结束的轨迹丢弃。

## 文件说明与局限

- `src/ptcg_rl/v2.py`：实际特征和顺序选择。
- `policy.py`：共享网络；其中旧版select_action不是V2入口。
- `train_v2.py`：在线训练；`export_v2.py`：schema校验及导出。
- `v3.py`保留为共享脚本兼容模块，V2命令显式指定v2；不是本版本实际使用的编码。

没有己方弃牌身份，无法区分Riptide所需的弃牌水能量；没有记忆、对手池和卡牌文本语义。场面只完整编码每方前5个备战位。可选/多选修复不等于战力已充分验证。

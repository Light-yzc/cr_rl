# 128局新种子复测最终结果与设备交接

2026-11-04（Asia/Shanghai）。前次中断保留120局，本次从Drive严格核对权重/代码/卡组/seed/seat、evaluation-to-audit SHA、每个回放SHA与原生结果，再跳过全部120局，仅执行原协议缺失的最后8局。远端184项测试通过。没有增加训练、换模型、调WAIT偏置或改变测试批量。

## 结果

- 完整128局：**76胜52负0平，得分率59.375%**。
- 16个固定卡组配对，每配对4组新模拟器seed、各2个策略换边局。
- 按deck pair聚类bootstrap的95%区间：**53.125%–65.625%**。
- 平均得分超过50%的配对8个，低于50%的1个，等于50%的7个；配对符号检验单侧p=**0.01953125**。
- 通过本次已固定复测的提升判据（区间下界>50%，且p≤.05）。

因此，本固定卡组/新种子复测支持**六轮PPO相对其准确BC初始化有改善**。16个配对包含重叠卡组，这个区间与检验是探索性证据，不能外推到所有卡组、高手或官方客户端。旧38局的区间跨50%是原本正确的小样本结论，保留不覆盖。

最后补测8局为5胜3负；前120局71胜49负。固定128局已全部结束，没有根据中途胜率继续追加样本。

## 完整性

- 全128局有效，128局原生重放parity全部通过。
- CPU挂载Drive后独立读回全部16块evaluation与执行审计，核对汇总文件与完成标记的SHA。
- 本地用相同注册协议重新调用聚合器，结果逐字段等于Drive最终summary。
- 最终summary SHA256：`e5a36ce149f68a0262afbbc968bcfee584705402e17145b1575d7c455eec9109`。
- policy SHA256：`2bf44455826ec00d7d58962ce3e07c93a93e0ca02bd053951ff5585b814bce54`。
- opponent SHA256：`ca7c91d1b54bb0cc326bfc04aff3652144de1bf8113deae20b78119c0326d1aa`。
- Drive：`MyDrive/cr_rl/runs/121m_ppo_seed_replication_20261104/`；有summary、completed、protocol、16块evaluation/原生审计/回放。
- 本地完整读回：`artifacts/ppo_seed_replication_20261104/final_readback.json`。
- 补测L4任务completed后按配置释放，服务器确认无GPU，只保留 `drive-reader-cpu`。

## 新设备

工作区快照、main历史bundle、todo和小型证据已打包为 `cr_rl_handoff_20261104.tar.gz`，同目录 `archive_receipt.json` 给出SHA与大小。包的Drive位置：`MyDrive/cr_rl/handoffs/20261104/`。解压后先读 `RESTORE.md`；恢复包含当前未提交修改，不复制旧设备venv、OAuth凭据或大型模型。

训练checkpoint仍是PPO generation6。继续训练要用完整checkpoint与opponent pool，不要拿Actor导出当resume。模型未扩容，当前原生122.62M；后续首要工作为CUDA BF16新旧概率对照和有预算的持续学习/遗忘曲线，之后才是增加卡组、长期池、300M对照和专精。官方客户端仍按用户要求延后，整个长期计划尚未完成。

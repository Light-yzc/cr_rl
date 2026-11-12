# 学习式落点读出的正式载入接口

正式实现位于 `src/cr_rl/models/learned_placement.py`，不导入 scripts。它严格兼容诊断阶段完整原头、宽残差、CNN、牌种表和牌种表+CNN的张量命名，尤其组合的 `original.original.*`、`original.card_location_bias.weight`、`spatial.*`。原头 `tile_grid_features` 为持久buffer；空间grid_xy由类重建，不作为payload张量。

## 独立decoder载入

```python
from cr_rl.models.learned_placement import attach_learned_placement

# trunk必须来自实际核过SHA的原受控checkpoint；不是同架构的任意新权重。
payload = torch.load(best_decoder_path, map_location="cpu", weights_only=False)
adapter = attach_learned_placement(
    model, payload, source_checkpoint_sha256=verified_trunk_file_sha256,
)
```

factory校验payload版本、原checkpoint SHA、完整model/temporal配置、决策间隔、variant及形状定义、无trunk/optimizer、heldout未用于选择、有限张量和参数总数；随后 `adapter.decoder.load_state_dict(..., strict=True)`。缺/多/错shape张量都拒绝。失败不注册模块，不改变原模型权重或memory generation。

载入完成只注册 `model.learned_placement` 与版本化definition，不偷偷monkeypatch旧方法。策略forward应明确分派，传入：

```text
context[B,temporal_dim]
encoded_entity_tokens[B,N,entity_dim]
entity_mask[B,N]
card_features[B,4,entity_dim]
entity_xy[B,N,2]  # 观测adapter之后、实体encoder之前的原归一化坐标
card_ids[B,4]    # 当前四张手牌原vocabulary ID，未知为-1
→ tile_logits[B,4,576]
```

推理不使用真值slot或真值卡。外部继续执行原合法mask、card softmax与条件tile softmax，所以卡牌/WAIT/技能概率质量不由新落点头改变。监督诊断另有 `forward_gt(..., slots[B], card_ids[B]或[B,4])`，保留原完整四slot attention查询路径。

payload可选 `temperature`，缺省1，必须有限且>0；完整落点logits在末端除以temperature，合法mask/归一化仍由外部处理。它不改变落点argmax，也不改变card/WAIT质量。任何温度选择必须只使用验证集，不能根据heldout异常调整。

## 完整模型checkpoint

definition格式为 `crrl-learned-placement-v1`，含variant、width、cnn_channels、alpha、维度、grid、输入契约、temperature。adapter还用 `_extra_state` 保存同一definition，加载不同版本/定义会拒绝。

完整模型checkpoint必须保存 `learned_placement_definition`。重建时先创建原StateActorCritic，再按definition注册 `PlacementAdapter`，最后对整个model `strict=True` 载入。不能先忽略新张量再strict=False加载，也不能丢失definition后猜测variant。root已在state_policy factory/forward接入此显式路径，相关CPU roundtrip覆盖遗漏definition拒绝。

## 参数与数值口径

当前183M配置：原模型182,795,519参数；prior+CNN64真正新增1,410,626。为了保留原控制支、兼容严格旧checkpoint，adapter另复制原落点头1,413,122，故adapter注册2,823,748，全模型注册185,619,267；实际forward使用的结构184,206,145。不能把有效新增量和注册总量混为一谈。

正式placement保持FP32参数和运算边界，trunk可用FP16 autocast；不支持对整个adapter直接 `.half()`。空间分支和原训练候选使用相同实体scatter、条件CNN和牌种可学习表，没有WAIT规则或高费保护。

19项小CPU测试覆盖：六variant真实诊断键名严载及非零残差输出一致、all4推理与GT路径一致、合法动作和WAIT质量、错误来源/配置/张量拒绝、版本化保存恢复、温度默认与正值校验。真实GPU拟合收益和全模型真实payload数值验收由单独实验报告说明，不能由这些CPU测试代替。

真实单行fixture `artifacts/bc_capacity_20261111/decoder_real_fixture_all_slots.pt` 含原缓存值及由源game0seat0流式定位的真实 `hand_card_ids=[[7,208,917,49]]`、完整合法mask、raw observation；GT卡身份和合法mask与原缓存精确匹配。它可供实际GPU载入后验证all4接口，无需合成手牌身份。

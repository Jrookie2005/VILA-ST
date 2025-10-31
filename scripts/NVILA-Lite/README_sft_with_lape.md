# VILA + LAPE SFT Training Script 使用说明

## 概述

`sft_with_lape.sh` 是基于原始 `sft.sh` 脚本修改的版本，增加了对 LAPE (Learnable Absolute Position Embeddings) 的支持。

## 使用方法

### 1. 基本用法（不启用LAPE）

```bash
./scripts/NVILA-Lite/sft_with_lape.sh
```

这将使用默认参数运行，LAPE 功能默认关闭。

### 2. 启用LAPE训练

```bash
ENABLE_LAPE=true ./scripts/NVILA-Lite/sft_with_lape.sh
```

### 3. 自定义LAPE参数

```bash
ENABLE_LAPE=true \
NUM_SPATIAL_TOKENS=150 \
NUM_TEMPORAL_TOKENS=200 \
./scripts/NVILA-Lite/sft_with_lape.sh
```

### 4. 指定路径参数

```bash
ENABLE_LAPE=true ./scripts/NVILA-Lite/sft_with_lape.sh \
  "runs/train/my-pretrain/model" \
  "my-data-mixture" \
  "runs/train/my-sft-output"
```

## 参数说明

### 位置参数

1. `STAGE_PATH` (可选): 预训练模型路径
   - 默认: `"runs/train/nvila-8b-pretrain/model"`

2. `DATA_MIXTURE` (可选): 训练数据配置
   - 默认: `"nvila-pretrain"`

3. `OUTPUT_DIR` (可选): 输出目录
   - 默认: `"runs/train/nvila-8b-sft"`

### LAPE 环境变量

- `ENABLE_LAPE`: 启用/禁用 LAPE 功能
  - 值: `true` 或 `false`
  - 默认: `false`

- `NUM_SPATIAL_TOKENS`: 空间位置token数量
  - 值: 正整数
  - 默认: `100`
  - 推荐范围: 50-200

- `NUM_TEMPORAL_TOKENS`: 时间位置token数量
  - 值: 正整数
  - 默认: `100`
  - 推荐范围: 50-200

## 脚本特性

### ✅ 完善的检查机制

1. **参数验证**: 检查LAPE参数是否为有效的正整数
2. **路径验证**: 验证输入路径和配置文件是否存在
3. **依赖检查**: 确认训练脚本和配置文件可用

### 📊 详细的进度信息

```
====================================================
🔧 VILA SFT with LAPE Configuration
====================================================
📂 Stage Path: runs/train/nvila-8b-pretrain/model
📊 Data Mixture: nvila-pretrain
💾 Output Dir: runs/train/nvila-8b-sft
🧠 LAPE Enabled: true
  🗺️  Spatial Tokens: 100
  ⏰ Temporal Tokens: 100
====================================================
```

### ⚠️ 内存警告

启用LAPE时会显示内存使用警告：

```
⚠️  WARNING: LAPE will increase memory usage due to additional embeddings
   Recommended: Monitor GPU memory and adjust batch size if needed
```

### 🎉 训练完成总结

训练完成后显示详细信息：

```
====================================================
🎉 Training Completed!
====================================================
📂 Model saved to: runs/train/nvila-8b-sft/model
🧠 LAPE was: ENABLED
  🗺️  Used 100 spatial tokens
  ⏰ Used 100 temporal tokens
====================================================
```

## 性能建议

### 内存优化

1. **启用LAPE时**:
   - 监控GPU内存使用
   - 如需要可减少 `per_device_train_batch_size`
   - 考虑调整 `gradient_accumulation_steps`

2. **Token数量选择**:
   - 较少token (50-100): 内存友好，基础位置信息
   - 较多token (150-200): 更精细位置编码，但内存需求更高

### 训练策略

1. **预训练阶段**: 建议启用LAPE获得最佳效果
2. **微调阶段**: 根据下游任务决定是否使用LAPE
3. **评估阶段**: 保持训练时的LAPE配置

## 故障排除

### 常见错误

1. **参数错误**:
   ```
   ❌ Error: NUM_SPATIAL_TOKENS must be a positive integer
   ```
   解决: 确保环境变量为正整数

2. **路径错误**:
   ```
   ❌ Error: Stage path does not exist: xxx
   ```
   解决: 检查预训练模型路径是否正确

3. **内存不足**:
   - 减少 `per_device_train_batch_size`
   - 降低LAPE token数量
   - 使用更大的GPU或更多GPU

### 调试建议

1. **启用详细日志**: 脚本已包含详细的配置输出
2. **监控资源**: 使用 `nvidia-smi` 监控GPU使用
3. **检查兼容性**: 确保VILA版本支持LAPE功能

## 相关文件

- `llava/train/train_mem.py`: 主训练脚本
- `llava/train/args.py`: 包含LAPE参数定义
- `llava/constants.py`: LAPE token定义
- `test_lape_simple.py`: LAPE功能测试脚本

---

**更新时间**: 2024年  
**兼容版本**: VILA + LAPE 集成版本
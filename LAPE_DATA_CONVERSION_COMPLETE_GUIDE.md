# LAPE 数据转换完整指南与示例

## 概述

本指南展示了如何将包含坐标和时间信息的训练数据转换为 LAPE (Learnable Absolute Position Embeddings) 格式，以便在 VILA+LAPE 模型中进行农业遥感训练。

## 转换的必要性

### 为什么需要转换？

1. **标准化表示**: 将不同格式的坐标统一转换为网格token
2. **位置嵌入**: LAPE 模块需要特定的token格式来学习空间-时间关系
3. **模型理解**: 让模型能够理解空间位置和时间序列信息
4. **跨图像一致性**: 不同尺寸图像映射到统一的12x12空间网格

## Token 格式说明

### 空间Token (12x12网格)
- **高度token**: `<HEIGHT-000>` 到 `<HEIGHT-011>` (共12个)
- **宽度token**: `<WIDTH-000>` 到 `<WIDTH-011>` (共12个)  
- **组合**: `<HEIGHT-002><WIDTH-003>` 表示网格位置 (2,3)

### 时间Token (64个时间点)
- **时间token**: `<TEMP-000>` 到 `<TEMP-063>` (共64个)
- **基准**: 2020年1月为 `<TEMP-000>`，每个token代表1个月
- **示例**: `<TEMP-042>` = 2023年7月

## 转换示例对比

### 示例1: 基础坐标转换

**原始数据**:
```json
{
  "from": "human",
  "value": "<image>分析坐标(120, 80)到(250, 200)区域的玉米作物健康状况。这张图片拍摄于2023年7月15日。"
}
```

**转换后**:
```json
{
  "from": "human", 
  "value": "<image>分析坐标<HEIGHT-002><WIDTH-003>到<HEIGHT-006><WIDTH-007>区域的玉米作物健康状况。这张图片拍摄于<TEMP-042>15日。"
}
```

**转换说明**:
- `(120, 80)` → `<HEIGHT-002><WIDTH-003>` (图像左上区域)
- `(250, 200)` → `<HEIGHT-006><WIDTH-007>` (图像中右区域)  
- `2023年7月` → `<TEMP-042>` (2020年1月后42个月)

### 示例2: 季节性比较

**原始数据**:
```json
{
  "from": "human",
  "value": "<image>比较春季和夏季期间，田地区域(100, 100)到(300, 300)的植被变化。"
}
```

**转换后**:
```json
{
  "from": "human",
  "value": "<image>比较<TEMP-038>和<TEMP-041>期间，田地区域<HEIGHT-003><WIDTH-003>到<HEIGHT-009><WIDTH-009>的植被变化。"
}
```

**转换说明**:
- `春季` → `<TEMP-038>` (2023年3月)
- `夏季` → `<TEMP-041>` (2023年6月)
- 田地区域坐标转换为网格表示

### 示例3: 时间序列监测

**原始数据**:
```json
{
  "from": "human",
  "value": "<image>监测从2023年5月到2023年9月期间，灌溉区域(50, 50, 200, 200)的效果。"
}
```

**转换后**:
```json
{
  "from": "human",
  "value": "<image>监测从<TEMP-040>到<TEMP-044>期间，灌溉区域<HEIGHT-001><WIDTH-001>到<HEIGHT-006><WIDTH-006>的效果。"
}
```

## 批量转换工具使用

### 基本使用方法

```bash
# 转换数据集
python batch_convert_to_lape_full.py input_dataset.json output_dataset_lape.json

# 转换并验证
python batch_convert_to_lape_full.py input_dataset.json output_dataset_lape.json --validate

# 自定义参数
python batch_convert_to_lape_full.py input_dataset.json output_dataset_lape.json \
    --max_spatial_tokens 144 \
    --max_temporal_tokens 64 \
    --image_size 384 384 \
    --base_date 2020-01-01 \
    --validate
```

### 实际转换效果

使用示例数据集的转换结果:

```
转换统计:
- 总样本数: 5
- 包含转换的样本: 5  
- 坐标转换次数: 18
- 时间转换次数: 18

Token使用统计:
- 使用的高度tokens: 7 个 (范围: 1-9)
- 使用的宽度tokens: 8 个 (范围: 1-9)  
- 使用的时间tokens: 6 个 (范围: 38-47)
```

## 支持的转换格式

### 坐标格式
1. **单点坐标**: `(x, y)` → `<HEIGHT-xxx><WIDTH-xxx>`
2. **边界框**: `(x1, y1, x2, y2)` → `<HEIGHT-xxx><WIDTH-xxx>到<HEIGHT-yyy><WIDTH-yyy>`
3. **中文描述**: `坐标(x,y)` → `位置<HEIGHT-xxx><WIDTH-xxx>`

### 时间格式
1. **标准日期**: `2023-07-15` → `<TEMP-042>`
2. **年月格式**: `2023年7月` → `<TEMP-042>`
3. **中文月份**: `七月` → `<TEMP-042>`
4. **中文季节**: `春季`, `夏季`, `秋季`, `冬季` → 对应的时间token
5. **英文季节**: `spring`, `summer`, `fall`, `winter` → 对应的时间token

## 农业应用场景

### 1. 作物监测
```json
{
  "original": "监测田地(100,100,300,300)在春季到夏季的玉米生长情况",
  "converted": "监测田地<HEIGHT-003><WIDTH-003>到<HEIGHT-009><WIDTH-009>在<TEMP-038>到<TEMP-041>的玉米生长情况"
}
```

### 2. 病虫害检测  
```json
{
  "original": "检查坐标(150,200)处2023年8月的病害斑点",
  "converted": "检查位置<HEIGHT-005><WIDTH-006>处<TEMP-043>的病害斑点"
}
```

### 3. 灌溉管理
```json
{
  "original": "分析灌溉区域(80,80,250,250)在夏季的水分状况", 
  "converted": "分析灌溉区域<HEIGHT-002><WIDTH-002>到<HEIGHT-007><WIDTH-007>在<TEMP-041>的水分状况"
}
```

### 4. 产量预测
```json
{
  "original": "预测农田(50,50,350,350)从春季到秋季的产量变化",
  "converted": "预测农田<HEIGHT-001><WIDTH-001>到<HEIGHT-010><WIDTH-010>从<TEMP-038>到<TEMP-044>的产量变化"
}
```

## 数据质量检查

### 验证转换结果
```python
# 自动验证脚本已包含在转换工具中
python batch_convert_to_lape_full.py data.json data_lape.json --validate

# 检查结果
✓ 所有LAPE tokens验证通过
✓ 数据集验证成功，可以用于LAPE训练
```

### 手动检查要点
1. **Token范围**: 空间token在0-11，时间token在0-63
2. **一致性**: 同一对话中的token使用一致
3. **语义保持**: 转换后文本语义清晰
4. **覆盖度**: 重要位置和时间信息都已转换

## 训练使用

转换后的数据可以直接用于VILA+LAPE训练:

```bash
# 使用转换后的LAPE数据集进行训练
bash scripts/agriculture_lape_warmup_stage1.sh \
    --data_path sample_agricultural_dataset_lape.json \
    --enable_lape \
    --lape_init_strategy agricultural
```

## 总结

通过将坐标和时间信息转换为LAPE token格式：

1. **提升模型理解**: 模型能够学习空间-时间关系
2. **统一表示**: 所有位置和时间信息标准化
3. **农业优化**: 特别适合农业遥感的时空分析
4. **易于扩展**: 支持各种坐标和时间格式

转换后的数据能够充分发挥LAPE模块在农业遥感任务中的优势，实现更准确的空间定位和时间序列建模。
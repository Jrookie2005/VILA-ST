# Agricultural LAPE-VILA Deployment Guide

## Overview
This guide provides step-by-step instructions for deploying VILA with LAPE (Learnable Absolute Position Embeddings) optimized for agricultural remote sensing applications.

## Prerequisites

### System Requirements
- GPU: NVIDIA GPU with >= 24GB VRAM (recommended: A100/H100)
- RAM: >= 64GB system RAM
- Storage: >= 500GB SSD for datasets and models
- CUDA: >= 11.8
- Python: >= 3.9

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd VILA

# Create conda environment
conda create -n vila-lape python=3.9
conda activate vila-lape

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## LAPE Configuration for Agriculture

### 1. Basic Agricultural Configuration
```python
# Model configuration for agricultural remote sensing
model_args = {
    # Enable LAPE
    "enable_lape": True,
    "lape_init_strategy": "agricultural",
    
    # Agricultural-optimized tokens
    "num_spatial_tokens": 144,      # 12x12 spatial grid for field coverage
    "num_temporal_tokens": 64,      # Seasonal/monthly temporal coverage
    
    # Learning rates optimized for satellite imagery
    "lape_spatial_lr": 1e-4,        # Higher for spatial features
    "lape_temporal_lr": 5e-5,       # Lower for temporal stability
    
    # Warmup strategy for agricultural domain
    "lape_warmup_steps": 1000,      # Progressive LAPE introduction
    "lape_freeze_warmup_steps": 300, # Initial freezing for stability
    
    # Position weights for agricultural focus
    "spatial_pos_weight": 1.2,      # Emphasize spatial relationships
    "temporal_pos_weight": 0.8,     # Moderate temporal emphasis
}
```

### 2. Training Stage Configuration

#### Stage 1: Agricultural Warmup (Steps 0-500)
```bash
# Use the agricultural warmup script
bash scripts/agriculture_lape_warmup_stage1.sh \
    --model_name_or_path Efficient-Large-Model/VILA1.5-3b \
    --data_path /path/to/agricultural/data.json \
    --image_folder /path/to/satellite/images \
    --vision_tower google/siglip-so400m-patch14-384 \
    --output_dir ./checkpoints/vila-lape-agriculture-stage1 \
    --num_train_epochs 1 \
    --enable_lape \
    --lape_init_strategy agricultural \
    --lape_warmup_steps 500 \
    --lape_freeze_warmup_steps 200
```

#### Stage 2: Full Agricultural Training (Steps 500+)
```bash
# Continue with full LAPE training
bash scripts/agriculture_lape_warmup_stage2.sh \
    --model_name_or_path ./checkpoints/vila-lape-agriculture-stage1 \
    --data_path /path/to/agricultural/data.json \
    --image_folder /path/to/satellite/images \
    --output_dir ./checkpoints/vila-lape-agriculture-final \
    --num_train_epochs 3 \
    --enable_lape \
    --lape_spatial_lr 1e-4 \
    --lape_temporal_lr 5e-5
```

## Dataset Preparation for Agriculture

### 1. Data Format
Prepare your agricultural remote sensing data in the following format:

```json
{
    "conversations": [
        {
            "from": "human",
            "value": "<image>What crop type is visible in this satellite image from <temporal_token_32>?"
        },
        {
            "from": "gpt", 
            "value": "This satellite image shows corn fields during the mid-growing season. The spatial patterns <spatial_height_token_8><spatial_width_token_12> indicate typical corn row spacing and the spectral characteristics suggest healthy crop development."
        }
    ],
    "image": "satellite_image_001.jpg",
    "spatial_context": {
        "height_tokens": 12,
        "width_tokens": 12,
        "resolution": "10m_per_pixel"
    },
    "temporal_context": {
        "temporal_tokens": 64,
        "acquisition_date": "2023-07-15",
        "season": "mid_growing"
    }
}
```

### 2. Spatial-Temporal Token Usage

#### Spatial Tokens
- Use for field boundaries, crop patterns, irrigation systems
- Range: `<spatial_height_token_0>` to `<spatial_height_token_11>`
- Range: `<spatial_width_token_0>` to `<spatial_width_token_11>`

#### Temporal Tokens  
- Use for seasonal changes, growth stages, harvest timing
- Range: `<temporal_token_0>` to `<temporal_token_63>`
- Map to specific dates or growth stages

### 3. Agricultural Domain Examples

```python
# Crop monitoring example
prompt = """
<image>
Analyze this satellite image acquired at <temporal_token_45> (late July).
Focus on the area marked by <spatial_height_token_6><spatial_width_token_8>.
What is the crop health status and estimated yield potential?
"""

# Irrigation management example  
prompt = """
<image>
This image shows irrigation patterns at <temporal_token_12> (early season).
Examine the spatial distribution <spatial_height_token_2><spatial_width_token_4> to <spatial_height_token_9><spatial_width_token_11>.
Are there any signs of water stress or irrigation inefficiencies?
"""

# Pest/disease detection example
prompt = """
<image>
Captured at <temporal_token_38> during peak growing season.
Investigate the anomalous area at <spatial_height_token_7><spatial_width_token_5>.
Could this pattern indicate pest damage or disease outbreak?
"""
```

## Performance Optimization

### 1. Memory Optimization
```python
# For large agricultural datasets
training_args = {
    "gradient_checkpointing": True,
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,
    "gradient_accumulation_steps": 8,
    "per_device_train_batch_size": 2,  # Adjust based on GPU memory
}
```

### 2. Multi-GPU Training
```bash
# Use DeepSpeed for large-scale agricultural training
torchrun --nproc_per_node=8 llava/train/train.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path Efficient-Large-Model/VILA1.5-3b \
    --data_path agricultural_dataset.json \
    --image_folder /path/to/satellite/images \
    --vision_tower google/siglip-so400m-patch14-384 \
    --output_dir ./checkpoints/vila-lape-agriculture-multi-gpu \
    --enable_lape \
    --lape_init_strategy agricultural \
    --num_train_epochs 5
```

## Inference and Deployment

### 1. Model Loading
```python
from llava.model.builder import load_pretrained_model

# Load agricultural LAPE-VILA model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="./checkpoints/vila-lape-agriculture-final",
    model_base=None,
    model_name="vila-lape-agriculture",
    load_8bit=False,
    load_4bit=False,
    device_map="auto"
)

# Verify LAPE is loaded
assert hasattr(model, 'enable_lape') and model.enable_lape
print("✓ Agricultural LAPE-VILA model loaded successfully")
```

### 2. Agricultural Inference Example
```python
import torch
from PIL import Image
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

def agricultural_inference(image_path, query, temporal_context=None, spatial_context=None):
    """
    Perform agricultural remote sensing inference with LAPE
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    images = process_images([image], image_processor, model.config)
    images = [img.to(device=model.device, dtype=torch.float16) for img in images]
    
    # Construct query with spatial-temporal tokens
    if temporal_context:
        query = query.replace("{temporal}", f"<temporal_token_{temporal_context}>")
    if spatial_context:
        for i, (h, w) in enumerate(spatial_context):
            query = query.replace(f"{{spatial_{i}}}", 
                                f"<spatial_height_token_{h}><spatial_width_token_{w}>")
    
    # Prepare conversation
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response.split("ASSISTANT:")[-1].strip()

# Example usage
result = agricultural_inference(
    image_path="satellite_field_001.jpg",
    query="<image>Analyze the crop health in this field acquired at {temporal}. Focus on areas {spatial_0} and {spatial_1}.",
    temporal_context=45,  # Late July
    spatial_context=[(3, 4), (7, 8)]  # Two field areas
)
print(result)
```

## Validation and Testing

### 1. Run Comprehensive Tests
```bash
# Test all LAPE functionality
python test_lape_comprehensive.py --verbose

# Test agricultural-specific features
python test_lape_integration.py --test-agricultural
```

### 2. Performance Benchmarking
```python
# Benchmark agricultural tasks
tasks = [
    "crop_classification",
    "yield_prediction", 
    "disease_detection",
    "irrigation_assessment",
    "harvest_timing"
]

for task in tasks:
    print(f"Benchmarking {task}...")
    # Run task-specific evaluation
    # Compare LAPE vs non-LAPE performance
```

## Troubleshooting

### Common Issues

1. **LAPE tokens not found**
   - Ensure `initialize_spatial_temporal_tokens()` is called
   - Check that constants are properly imported

2. **Memory issues with large images**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Poor agricultural domain performance**
   - Verify agricultural initialization strategy
   - Check spatial/temporal token usage in data
   - Adjust learning rates for domain

### Debug Commands
```bash
# Check LAPE integration
python -c "
from llava.model.builder import load_pretrained_model
model = load_pretrained_model('model_path')[1]
print('LAPE enabled:', hasattr(model, 'enable_lape') and model.enable_lape)
"

# Verify agricultural tokens
python -c "
from llava.constants import *
tokens = [TEMPORAL_INPUT_TOKEN, SPATIAL_HEIGHT_INPUT_TOKEN, SPATIAL_WIDTH_INPUT_TOKEN]
print('Agricultural tokens:', tokens)
"
```

## Best Practices

1. **Data Preparation**
   - Ensure balanced spatial-temporal token distribution
   - Include diverse agricultural scenarios
   - Validate token usage consistency

2. **Training Strategy**
   - Start with agricultural warmup
   - Monitor LAPE component gradients
   - Use domain-specific validation metrics

3. **Inference Optimization**
   - Cache processed images when possible
   - Use appropriate temperature settings
   - Batch similar spatial-temporal queries

4. **Monitoring**
   - Track spatial vs temporal attention patterns
   - Monitor agricultural task-specific metrics
   - Log LAPE component activations

## Conclusion

This deployment guide provides a comprehensive framework for using VILA with LAPE in agricultural remote sensing applications. The agricultural optimization ensures effective spatial-temporal understanding for satellite imagery analysis, crop monitoring, and precision agriculture tasks.

For additional support, refer to the LAPE integration tests and agricultural analysis documentation.
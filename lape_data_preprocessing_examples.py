#!/usr/bin/env python3
"""
LAPE Data Preprocessing Examples
Shows how to convert coordinates and temporal information to LAPE tokens
"""

import json
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

# LAPE token formats (consistent with constants.py)
TEMPORAL_TOKEN_FORMAT = '<TEMP-{:03d}>'
SPATIAL_HEIGHT_TOKEN_FORMAT = '<HEIGHT-{:03d}>'
SPATIAL_WIDTH_TOKEN_FORMAT = '<WIDTH-{:03d}>'

class LAPEDataPreprocessor:
    """Preprocessor for converting coordinates and temporal info to LAPE tokens"""
    
    def __init__(self, 
                 max_spatial_tokens: int = 144,  # 12x12 grid for agriculture
                 max_temporal_tokens: int = 64,   # ~5 years of monthly data
                 image_size: Tuple[int, int] = (384, 384),
                 temporal_range_years: int = 5):
        self.max_spatial_tokens = max_spatial_tokens
        self.max_temporal_tokens = max_temporal_tokens
        self.image_size = image_size
        self.temporal_range_years = temporal_range_years
        
        # Calculate spatial grid dimensions
        self.spatial_grid_size = int(max_spatial_tokens ** 0.5)  # 12x12 = 144
        
    def convert_pixel_coordinates_to_tokens(self, x: int, y: int, 
                                          image_width: int = None, 
                                          image_height: int = None) -> Tuple[str, str]:
        """
        Convert pixel coordinates to spatial tokens
        
        Args:
            x, y: pixel coordinates
            image_width, image_height: image dimensions (use default if None)
            
        Returns:
            Tuple of (height_token, width_token)
        """
        if image_width is None:
            image_width = self.image_size[0]
        if image_height is None:
            image_height = self.image_size[1]
            
        # Normalize coordinates to grid
        grid_x = int((x / image_width) * self.spatial_grid_size)
        grid_y = int((y / image_height) * self.spatial_grid_size)
        
        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.spatial_grid_size - 1))
        grid_y = max(0, min(grid_y, self.spatial_grid_size - 1))
        
        height_token = SPATIAL_HEIGHT_TOKEN_FORMAT.format(grid_y)
        width_token = SPATIAL_WIDTH_TOKEN_FORMAT.format(grid_x)
        
        return height_token, width_token
    
    def convert_bbox_to_tokens(self, bbox: List[int], 
                              image_width: int = None, 
                              image_height: int = None) -> Dict[str, str]:
        """
        Convert bounding box to spatial tokens
        
        Args:
            bbox: [x1, y1, x2, y2] format
            
        Returns:
            Dictionary with start and end tokens
        """
        x1, y1, x2, y2 = bbox
        
        start_h, start_w = self.convert_pixel_coordinates_to_tokens(
            x1, y1, image_width, image_height)
        end_h, end_w = self.convert_pixel_coordinates_to_tokens(
            x2, y2, image_width, image_height)
        
        return {
            'start_height': start_h,
            'start_width': start_w,
            'end_height': end_h,
            'end_width': end_w
        }
    
    def convert_date_to_temporal_token(self, date_str: str, 
                                     base_date: str = "2020-01-01") -> str:
        """
        Convert date string to temporal token
        
        Args:
            date_str: Date in YYYY-MM-DD format
            base_date: Reference date for temporal indexing
            
        Returns:
            Temporal token string
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            base = datetime.strptime(base_date, "%Y-%m-%d")
            
            # Calculate months since base date
            months_diff = (target_date.year - base.year) * 12 + (target_date.month - base.month)
            
            # Clamp to valid range
            temporal_index = max(0, min(months_diff, self.max_temporal_tokens - 1))
            
            return TEMPORAL_TOKEN_FORMAT.format(temporal_index)
            
        except ValueError:
            # Fallback for invalid dates
            return TEMPORAL_TOKEN_FORMAT.format(0)
    
    def convert_season_to_temporal_token(self, season: str, year: int = 2023) -> str:
        """
        Convert season to temporal token
        
        Args:
            season: "spring", "summer", "fall", "winter"
            year: Year for the season
            
        Returns:
            Temporal token string
        """
        season_map = {
            "spring": 3,   # March
            "summer": 6,   # June  
            "fall": 9,     # September
            "autumn": 9,   # September (alias)
            "winter": 12   # December
        }
        
        month = season_map.get(season.lower(), 6)  # Default to summer
        date_str = f"{year}-{month:02d}-01"
        
        return self.convert_date_to_temporal_token(date_str)

def create_agricultural_training_examples():
    """Create training examples for agricultural remote sensing"""
    
    processor = LAPEDataPreprocessor()
    examples = []
    
    # Example 1: Crop field analysis with spatial coordinates
    example1 = {
        "conversations": [
            {
                "from": "human", 
                "value": "<image>Analyze the crop health in the field area from coordinates (120, 80) to (250, 200). This image was taken in July 2023."
            },
            {
                "from": "gpt",
                "value": "Based on the satellite image, I can analyze the specified field area. The crops in this region show healthy growth patterns with good vegetation density."
            }
        ],
        "image": "satellite_field_001.jpg"
    }
    
    # Convert coordinates to LAPE tokens
    bbox_tokens = processor.convert_bbox_to_tokens([120, 80, 250, 200])
    temporal_token = processor.convert_date_to_temporal_token("2023-07-15")
    
    # Replace with LAPE tokens
    example1_converted = {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>Analyze the crop health in the field area from {bbox_tokens['start_height']}{bbox_tokens['start_width']} to {bbox_tokens['end_height']}{bbox_tokens['end_width']}. This image was taken at {temporal_token}."
            },
            {
                "from": "gpt", 
                "value": f"Based on the satellite image acquired at {temporal_token}, I can analyze the specified field area {bbox_tokens['start_height']}{bbox_tokens['start_width']} to {bbox_tokens['end_height']}{bbox_tokens['end_width']}. The crops in this region show healthy growth patterns with good vegetation density."
            }
        ],
        "image": "satellite_field_001.jpg",
        "lape_metadata": {
            "spatial_tokens": bbox_tokens,
            "temporal_token": temporal_token,
            "original_bbox": [120, 80, 250, 200],
            "original_date": "2023-07-15"
        }
    }
    
    examples.append(("Basic Agricultural Analysis", example1, example1_converted))
    
    # Example 2: Multi-region comparison with seasons
    example2 = {
        "conversations": [
            {
                "from": "human",
                "value": "<image>Compare the vegetation between region A (50, 50, 150, 150) and region B (200, 200, 300, 300) during spring season."
            },
            {
                "from": "gpt", 
                "value": "I'll compare the vegetation in both regions. Region A shows early spring growth while region B has more advanced vegetation development."
            }
        ],
        "image": "multi_region_comparison.jpg"
    }
    
    # Convert to LAPE format
    region_a_tokens = processor.convert_bbox_to_tokens([50, 50, 150, 150])
    region_b_tokens = processor.convert_bbox_to_tokens([200, 200, 300, 300])
    spring_token = processor.convert_season_to_temporal_token("spring", 2023)
    
    example2_converted = {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>Compare the vegetation between region A ({region_a_tokens['start_height']}{region_a_tokens['start_width']} to {region_a_tokens['end_height']}{region_a_tokens['end_width']}) and region B ({region_b_tokens['start_height']}{region_b_tokens['start_width']} to {region_b_tokens['end_height']}{region_b_tokens['end_width']}) during {spring_token}."
            },
            {
                "from": "gpt",
                "value": f"I'll compare the vegetation in both regions at {spring_token}. Region A ({region_a_tokens['start_height']}{region_a_tokens['start_width']} to {region_a_tokens['end_height']}{region_a_tokens['end_width']}) shows early spring growth while region B ({region_b_tokens['start_height']}{region_b_tokens['start_width']} to {region_b_tokens['end_height']}{region_b_tokens['end_width']}) has more advanced vegetation development."
            }
        ],
        "image": "multi_region_comparison.jpg",
        "lape_metadata": {
            "region_a_tokens": region_a_tokens,
            "region_b_tokens": region_b_tokens,
            "temporal_token": spring_token,
            "season": "spring"
        }
    }
    
    examples.append(("Multi-Region Comparison", example2, example2_converted))
    
    # Example 3: Temporal progression analysis
    example3 = {
        "conversations": [
            {
                "from": "human",
                "value": "<image>Track the changes in this field (100, 100, 280, 280) from March 2023 to September 2023."
            },
            {
                "from": "gpt",
                "value": "Analyzing the temporal changes from early growing season to harvest time shows significant crop development and maturation."
            }
        ],
        "image": "temporal_sequence.jpg"
    }
    
    # Convert temporal range
    field_tokens = processor.convert_bbox_to_tokens([100, 100, 280, 280])
    march_token = processor.convert_date_to_temporal_token("2023-03-01")
    september_token = processor.convert_date_to_temporal_token("2023-09-01")
    
    example3_converted = {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>Track the changes in this field ({field_tokens['start_height']}{field_tokens['start_width']} to {field_tokens['end_height']}{field_tokens['end_width']}) from {march_token} to {september_token}."
            },
            {
                "from": "gpt",
                "value": f"Analyzing the temporal changes from {march_token} to {september_token} in field area {field_tokens['start_height']}{field_tokens['start_width']} to {field_tokens['end_height']}{field_tokens['end_width']} shows significant crop development and maturation."
            }
        ],
        "image": "temporal_sequence.jpg",
        "lape_metadata": {
            "field_tokens": field_tokens,
            "start_temporal": march_token,
            "end_temporal": september_token,
            "temporal_range": ["2023-03-01", "2023-09-01"]
        }
    }
    
    examples.append(("Temporal Progression", example3, example3_converted))
    
    return examples

def create_batch_conversion_script():
    """Create script for batch converting existing datasets"""
    
    script_content = '''
def convert_dataset_to_lape_format(input_file: str, output_file: str):
    """
    Convert existing dataset to LAPE token format
    """
    processor = LAPEDataPreprocessor()
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        converted_item = convert_single_item(item, processor)
        converted_data.append(converted_item)
    
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

def convert_single_item(item: Dict, processor: LAPEDataPreprocessor) -> Dict:
    """Convert a single data item to LAPE format"""
    
    converted_item = item.copy()
    
    for i, conv in enumerate(item.get('conversations', [])):
        if 'value' in conv:
            # Convert coordinates in format (x1, y1, x2, y2)
            value = conv['value']
            value = convert_coordinates_in_text(value, processor)
            value = convert_dates_in_text(value, processor)
            value = convert_seasons_in_text(value, processor)
            
            converted_item['conversations'][i]['value'] = value
    
    return converted_item

def convert_coordinates_in_text(text: str, processor: LAPEDataPreprocessor) -> str:
    """Convert coordinate patterns in text to LAPE tokens"""
    
    # Pattern for (x1, y1, x2, y2) format
    bbox_pattern = r'\\((\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\)'
    
    def replace_bbox(match):
        x1, y1, x2, y2 = map(int, match.groups())
        tokens = processor.convert_bbox_to_tokens([x1, y1, x2, y2])
        return f"{tokens['start_height']}{tokens['start_width']} to {tokens['end_height']}{tokens['end_width']}"
    
    text = re.sub(bbox_pattern, replace_bbox, text)
    
    # Pattern for single coordinates (x, y)
    coord_pattern = r'\\((\\d+),\\s*(\\d+)\\)'
    
    def replace_coord(match):
        x, y = map(int, match.groups())
        h_token, w_token = processor.convert_pixel_coordinates_to_tokens(x, y)
        return f"{h_token}{w_token}"
    
    text = re.sub(coord_pattern, replace_coord, text)
    
    return text

def convert_dates_in_text(text: str, processor: LAPEDataPreprocessor) -> str:
    """Convert date patterns to temporal tokens"""
    
    # Pattern for YYYY-MM-DD format
    date_pattern = r'(\\d{4}-\\d{2}-\\d{2})'
    
    def replace_date(match):
        date_str = match.group(1)
        return processor.convert_date_to_temporal_token(date_str)
    
    text = re.sub(date_pattern, replace_date, text)
    
    # Pattern for "Month YYYY" format
    month_year_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})'
    
    def replace_month_year(match):
        month_name, year = match.groups()
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month = month_map.get(month_name, 6)
        date_str = f"{year}-{month:02d}-01"
        return processor.convert_date_to_temporal_token(date_str)
    
    text = re.sub(month_year_pattern, replace_month_year, text)
    
    return text

def convert_seasons_in_text(text: str, processor: LAPEDataPreprocessor) -> str:
    """Convert season references to temporal tokens"""
    
    # Pattern for "season YYYY" format
    season_pattern = r'(spring|summer|fall|autumn|winter)\\s+(\\d{4})'
    
    def replace_season(match):
        season, year = match.groups()
        return processor.convert_season_to_temporal_token(season, int(year))
    
    text = re.sub(season_pattern, replace_season, text, flags=re.IGNORECASE)
    
    return text
'''
    
    return script_content

def demonstrate_token_usage():
    """Demonstrate proper LAPE token usage patterns"""
    
    print("LAPE Token Usage Patterns for Agricultural Data")
    print("=" * 60)
    
    processor = LAPEDataPreprocessor()
    
    # 1. Spatial Token Examples
    print("\n1. Spatial Token Examples:")
    print("-" * 30)
    
    # Different coordinate formats
    coords_examples = [
        (120, 80, "Center of field"),
        (50, 50, "Top-left corner"),
        (350, 350, "Bottom-right corner"),
        ([100, 100, 200, 200], "Bounding box")
    ]
    
    for coords in coords_examples:
        if isinstance(coords[0], list):
            bbox_tokens = processor.convert_bbox_to_tokens(coords[0])
            print(f"{coords[1]}: {bbox_tokens['start_height']}{bbox_tokens['start_width']} to {bbox_tokens['end_height']}{bbox_tokens['end_width']}")
        else:
            h_token, w_token = processor.convert_pixel_coordinates_to_tokens(coords[0], coords[1])
            print(f"{coords[2]}: {h_token}{w_token}")
    
    # 2. Temporal Token Examples  
    print("\n2. Temporal Token Examples:")
    print("-" * 30)
    
    temporal_examples = [
        ("2023-01-15", "Winter planting"),
        ("2023-04-01", "Spring growth"),
        ("2023-07-15", "Mid-season development"),
        ("2023-10-01", "Harvest time"),
        ("spring", "Spring season"),
        ("summer", "Summer season")
    ]
    
    for date_or_season, description in temporal_examples:
        if "-" in date_or_season:
            token = processor.convert_date_to_temporal_token(date_or_season)
        else:
            token = processor.convert_season_to_temporal_token(date_or_season, 2023)
        print(f"{description}: {token}")
    
    # 3. Combined Usage Examples
    print("\n3. Combined Spatial-Temporal Usage:")
    print("-" * 40)
    
    combined_examples = [
        {
            "description": "Field monitoring over time",
            "spatial": [100, 100, 200, 200],
            "temporal": "2023-06-15",
            "query": "Monitor crop growth in field area {spatial} during {temporal}"
        },
        {
            "description": "Multi-season comparison",
            "spatial": [150, 150, 250, 250],
            "temporal": ["spring", "summer"],
            "query": "Compare vegetation in area {spatial} between {temporal[0]} and {temporal[1]}"
        }
    ]
    
    for example in combined_examples:
        print(f"\n{example['description']}:")
        
        bbox_tokens = processor.convert_bbox_to_tokens(example['spatial'])
        spatial_str = f"{bbox_tokens['start_height']}{bbox_tokens['start_width']} to {bbox_tokens['end_height']}{bbox_tokens['end_width']}"
        
        if isinstance(example['temporal'], list):
            temporal_tokens = [processor.convert_season_to_temporal_token(season, 2023) for season in example['temporal']]
            temporal_str = f"{temporal_tokens[0]} and {temporal_tokens[1]}"
        else:
            if "-" in example['temporal']:
                temporal_str = processor.convert_date_to_temporal_token(example['temporal'])
            else:
                temporal_str = processor.convert_season_to_temporal_token(example['temporal'], 2023)
        
        query = example['query'].format(spatial=spatial_str, temporal=temporal_str)
        print(f"Query: {query}")

if __name__ == "__main__":
    print("LAPE Data Preprocessing Examples")
    print("=" * 50)
    
    # Create examples
    examples = create_agricultural_training_examples()
    
    for name, original, converted in examples:
        print(f"\n{name}")
        print("-" * len(name))
        print("Original:")
        print(json.dumps(original, indent=2))
        print("\nConverted to LAPE:")
        print(json.dumps(converted, indent=2))
        print("\n" + "="*50)
    
    # Demonstrate token usage
    demonstrate_token_usage()
    
    # Save batch conversion script
    with open('batch_convert_to_lape.py', 'w') as f:
        f.write(create_batch_conversion_script())
    
    print("\nBatch conversion script saved as 'batch_convert_to_lape.py'")
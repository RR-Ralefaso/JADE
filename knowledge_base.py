import numpy as np
import colorsys
import cv2
from typing import Dict, List, Tuple, Optional
import os

# Simple dictionary-based knowledge base (no pickle)
ENHANCED_KNOWLEDGE = {
    "person": {
        "info": "Human being. Most common object in COCO dataset.",
        "value": "Priceless. People are the most valuable asset.",
        "tip": "Track multiple people with unique IDs for crowd analysis.",
        "brands": ["N/A", "Human"],
        "conditions": ["active", "stationary", "resting"],
        "materials": ["clothing_varied"],
        "estimated_value": "Priceless",
        "maintenance": "Proper nutrition, exercise, healthcare",
        "rarity": "Common",
        "authenticity_indicators": ["normal proportions", "natural movement"],
        "category": "human",
        "base_value": 0
    },
    
    "car": {
        "info": "Four-wheeled motor vehicle. Most common personal transport.",
        "value": "Check make/model/year. Luxury brands = premium value.",
        "tip": "Low mileage + clean title = best resale value.",
        "brands": ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Tesla"],
        "conditions": ["showroom", "excellent", "good", "fair", "poor", "salvage"],
        "materials": ["steel", "aluminum", "carbon_fiber"],
        "estimated_value": "$5,000 - $300,000+",
        "maintenance": "Regular oil changes, tire rotation, brake service",
        "rarity": "Common to Rare (depending on model)",
        "authenticity_indicators": ["VIN number", "original parts", "service records"],
        "category": "vehicle",
        "base_value": 20000
    },
    
    "laptop": {
        "info": "Portable computer for work/entertainment.",
        "value": "Apple MacBooks hold value best.",
        "tip": "Check battery cycle count for MacBooks.",
        "brands": ["Apple", "Dell", "HP", "Lenovo", "Microsoft", "Asus"],
        "conditions": ["sealed", "like_new", "used", "refurbished", "broken"],
        "materials": ["aluminum", "plastic", "magnesium_alloy"],
        "estimated_value": "$200 - $5,000",
        "maintenance": "Regular updates, clean vents, proper charging",
        "rarity": "Common",
        "authenticity_indicators": ["serial numbers", "OS authenticity", "build quality"],
        "category": "electronics",
        "base_value": 800
    },
    
    "cell phone": {
        "info": "Mobile communication device.",
        "value": "Latest iPhones/Samsungs = highest resale.",
        "tip": "Check IMEI for blacklist status.",
        "brands": ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"],
        "conditions": ["sealed", "mint", "good", "fair", "broken"],
        "materials": ["glass", "aluminum", "plastic"],
        "estimated_value": "$50 - $1,500",
        "maintenance": "Use case, avoid extreme temps, proper charging",
        "rarity": "Common",
        "authenticity_indicators": ["IMEI check", "original box", "software authenticity"],
        "category": "electronics",
        "base_value": 500
    },
    
    "bottle": {
        "info": "Glass/plastic liquid container.",
        "value": "Vintage glass bottles collectible.",
        "tip": "Recyclable materials increasingly valuable.",
        "brands": ["Hydro Flask", "Yeti", "Nalgene", "CamelBak", "S'well"],
        "conditions": ["new", "used", "scratched", "damaged"],
        "materials": ["stainless_steel", "plastic", "glass", "aluminum"],
        "estimated_value": "$5 - $100",
        "maintenance": "Regular cleaning, avoid drops",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "material quality", "weight"],
        "category": "container",
        "base_value": 25
    },
    
    "chair": {
        "info": "Furniture for sitting.",
        "value": "Designer/antique chairs = high value.",
        "tip": "Check joints and upholstery condition.",
        "brands": ["Herman Miller", "Steelcase", "IKEA", "Eames", "Knoll"],
        "conditions": ["new", "excellent", "good", "fair", "poor"],
        "materials": ["wood", "metal", "plastic", "fabric", "leather"],
        "estimated_value": "$20 - $5,000+",
        "maintenance": "Clean regularly, tighten joints, condition materials",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["brand markings", "construction quality", "material authenticity"],
        "category": "furniture",
        "base_value": 150
    },
    
    "book": {
        "info": "Printed or written work.",
        "value": "First editions/signed copies = premium.",
        "tip": "Check for signatures and edition numbers.",
        "brands": ["Various publishers"],
        "conditions": ["new", "like_new", "used", "worn", "collectible"],
        "materials": ["paper", "leather", "cloth"],
        "estimated_value": "$1 - $10,000+",
        "maintenance": "Store upright, avoid moisture/sunlight",
        "rarity": "Common to Ultra Rare",
        "authenticity_indicators": ["ISBN", "printing details", "signature verification"],
        "category": "literature",
        "base_value": 20
    },
    
    "tv": {
        "info": "Television for entertainment.",
        "value": "Vintage TVs and high-end models hold value.",
        "tip": "Check screen condition and functionality.",
        "brands": ["Samsung", "LG", "Sony", "Panasonic", "Vintage RCA"],
        "conditions": ["new", "used", "vintage", "broken", "refurbished"],
        "materials": ["plastic", "glass", "metal", "electronics"],
        "estimated_value": "$50 - $10,000",
        "maintenance": "Clean screen properly, ensure ventilation, update software",
        "rarity": "Common",
        "authenticity_indicators": ["model numbers", "brand logos", "functionality"],
        "category": "electronics",
        "base_value": 500
    },
    
    "backpack": {
        "info": "Carried bag with straps. School/travel essential.",
        "value": "Brand name (Patagonia, North Face) = higher resale.",
        "tip": "Check zippers + fabric condition.",
        "brands": ["Patagonia", "North Face", "Jansport", "Osprey", "Herschel"],
        "conditions": ["new", "excellent", "good", "worn", "damaged"],
        "materials": ["nylon", "polyester", "canvas", "leather"],
        "estimated_value": "$20 - $500",
        "maintenance": "Spot clean, avoid overloading, store empty",
        "rarity": "Common",
        "authenticity_indicators": ["brand tags", "quality stitching", "material feel"],
        "category": "accessory",
        "base_value": 50
    },
    
    "bicycle": {
        "info": "Two-wheeled pedal vehicle. Eco-friendly transport.",
        "value": "Check brand: Trek, Specialized = higher resale.",
        "tip": "Carbon fiber frames are lighter but more expensive.",
        "brands": ["Trek", "Specialized", "Giant", "Cannondale", "Santa Cruz"],
        "conditions": ["new", "like_new", "used", "needs_repair", "vintage"],
        "materials": ["aluminum", "carbon_fiber", "steel", "titanium"],
        "estimated_value": "$200 - $12,000",
        "maintenance": "Chain lubrication, brake adjustment, tire pressure",
        "rarity": "Common",
        "authenticity_indicators": ["brand logos", "serial numbers", "quality welds"],
        "category": "vehicle",
        "base_value": 300
    },
    
    "wine glass": {
        "info": "Drinking glass for wine. Can indicate lifestyle or event.",
        "value": "Crystal glassware = premium value.",
        "tip": "Check for lead content in crystal glassware.",
        "brands": ["Waterford", "Baccarat", "Riedel", "Spiegelau"],
        "conditions": ["new", "used", "vintage", "cracked", "chipped"],
        "materials": ["crystal", "glass", "lead_crystal"],
        "estimated_value": "$10 - $1,000+",
        "maintenance": "Hand wash, avoid extreme temperatures",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["brand etching", "clarity", "ring sound"],
        "category": "tableware",
        "base_value": 50
    },
    
    "vase": {
        "info": "Container for holding flowers or as decorative item.",
        "value": "Antique/designer vases = high value.",
        "tip": "Check for signatures and manufacturing marks.",
        "brands": ["Royal Copenhagen", "Wedgwood", "Meissen", "Ming Dynasty"],
        "conditions": ["antique", "vintage", "modern", "cracked", "restored"],
        "materials": ["porcelain", "ceramic", "glass", "crystal"],
        "estimated_value": "$20 - $50,000+",
        "maintenance": "Dust regularly, avoid direct sunlight",
        "rarity": "Common to Ultra Rare",
        "authenticity_indicators": ["maker's mark", "age signs", "material quality"],
        "category": "decor",
        "base_value": 100
    },
    
    "cup": {
        "info": "Drinking vessel, often with handle.",
        "value": "Collectible mugs and antique teacups hold value.",
        "tip": "Check for manufacturer stamps on bottom.",
        "brands": ["Starbucks", "Wedgwood", "Royal Albert", "Narumi"],
        "conditions": ["new", "used", "vintage", "cracked", "stained"],
        "materials": ["porcelain", "ceramic", "glass", "stoneware"],
        "estimated_value": "$5 - $500",
        "maintenance": "Hand wash recommended for delicate pieces",
        "rarity": "Common",
        "authenticity_indicators": ["maker's mark", "glaze quality", "weight"],
        "category": "tableware",
        "base_value": 15
    },
    
    "keyboard": {
        "info": "Computer input device or musical instrument.",
        "value": "Mechanical keyboards and vintage synthesizers = high value.",
        "tip": "Check switch type for mechanical keyboards.",
        "brands": ["Corsair", "Logitech", "Razer", "Moog", "Yamaha"],
        "conditions": ["new", "used", "vintage", "mechanical", "membrane"],
        "materials": ["plastic", "metal", "electronics"],
        "estimated_value": "$20 - $5,000",
        "maintenance": "Regular cleaning, keycap removal for deep clean",
        "rarity": "Common",
        "authenticity_indicators": ["brand logos", "build quality", "key feel"],
        "category": "electronics",
        "base_value": 80
    },
    
    "mouse": {
        "info": "Computer pointing device.",
        "value": "Gaming mice and vintage models hold value.",
        "tip": "Check DPI and sensor type for gaming mice.",
        "brands": ["Logitech", "Razer", "SteelSeries", "Microsoft"],
        "conditions": ["new", "used", "gaming", "wireless", "wired"],
        "materials": ["plastic", "rubber", "electronics"],
        "estimated_value": "$10 - $200",
        "maintenance": "Clean sensor regularly, replace feet when worn",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "software compatibility", "sensor performance"],
        "category": "electronics",
        "base_value": 40
    }
}

# Material value multipliers
MATERIAL_VALUE_IMPACT = {
    "gold": 5.0,
    "platinum": 4.5,
    "diamond": 4.0,
    "carbon_fiber": 3.0,
    "titanium": 2.5,
    "leather": 2.0,
    "silver": 1.8,
    "stainless_steel": 1.5,
    "aluminum": 1.3,
    "glass": 1.2,
    "porcelain": 1.5,
    "wood": 1.0,
    "plastic": 0.8,
    "paper": 0.5
}

# Condition multipliers
CONDITION_MULTIPLIERS = {
    'excellent': 1.0,
    'good': 0.7,
    'fair': 0.4,
    'poor': 0.2,
    'unknown': 0.5
}

# Rarity multipliers
RARITY_MULTIPLIERS = {
    'ultra_rare': 10.0,
    'rare': 5.0,
    'uncommon': 2.0,
    'common': 1.0,
    'protected': 0.0
}

def analyze_object_visual_features(frame, bbox, object_type):
    """Analyze visual features of an object"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure bbox is within frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    
    if x2 <= x1 or y2 <= y1:
        return {}
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {}
    
    features = {}
    
    # Color analysis
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pixels = roi_rgb.reshape(-1, 3)
    
    if len(pixels) > 0:
        # Average color
        avg_color = np.mean(pixels, axis=0).astype(int)
        features['average_color'] = avg_color.tolist()
        features['color_name'] = _rgb_to_color_name(avg_color)
        
        # Color variance (for condition assessment)
        color_var = np.var(pixels, axis=0).mean()
        features['color_variance'] = float(color_var)
    
    # Texture analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    features['edge_density'] = float(edge_density)
    
    # Brightness and contrast
    features['brightness'] = float(np.mean(gray))
    features['contrast'] = float(np.std(gray))
    
    # Sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['sharpness'] = float(laplacian_var)
    
    # Size info
    features['relative_size'] = roi.size / frame.size
    features['aspect_ratio'] = roi.shape[1] / roi.shape[0] if roi.shape[0] > 0 else 0
    
    # Material indicators
    features['material_indicators'] = _analyze_material_indicators(roi)
    
    # Condition indicators
    features['condition_indicators'] = _analyze_condition_indicators(roi, object_type)
    
    return features

def analyze_object_deep_features(frame, bbox, object_type):
    """Deep analysis of object features including detailed condition assessment"""
    features = analyze_object_visual_features(frame, bbox, object_type)
    
    if not features:
        return features
    
    # Add deep condition analysis
    if 'condition_indicators' in features:
        cond = features['condition_indicators']
        
        # Enhanced condition scoring
        condition_score = cond.get('condition_score', 5)
        
        # Analyze color consistency for wear
        if 'color_variance' in features:
            color_var = features['color_variance']
            if color_var > 5000:
                condition_score -= 1  # High variance indicates wear
            elif color_var < 1000:
                condition_score += 0.5  # Low variance indicates good condition
        
        # Analyze sharpness for focus/clarity
        if 'sharpness' in features:
            sharpness = features['sharpness']
            if sharpness < 100:
                condition_score -= 0.5  # Blurry indicates poor condition
        
        # Analyze edge density for texture/surface quality
        if 'edge_density' in features:
            edge_density = features['edge_density']
            if edge_density > 0.2:
                condition_score += 0.3  # Good texture detail
            elif edge_density < 0.05:
                condition_score -= 0.3  # Too smooth or blurry
        
        # Update condition
        cond['condition_score'] = max(0, min(10, condition_score))
        
        # Enhanced condition rating
        if condition_score >= 9:
            cond['overall_condition'] = 'excellent'
            cond['description'] = 'Like new, minimal wear'
            cond['recommendation'] = 'Excellent condition, ready for use or resale'
        elif condition_score >= 7:
            cond['overall_condition'] = 'good'
            cond['description'] = 'Minor wear, fully functional'
            cond['recommendation'] = 'Good condition, suitable for daily use'
        elif condition_score >= 5:
            cond['overall_condition'] = 'fair'
            cond['description'] = 'Visible wear, functional'
            cond['recommendation'] = 'Fair condition, may need maintenance'
        else:
            cond['overall_condition'] = 'poor'
            cond['description'] = 'Significant wear/damage'
            cond['recommendation'] = 'Poor condition, consider repair or replacement'
    
    # Add material confidence
    if 'material_indicators' in features:
        mat = features['material_indicators']
        if mat.get('reflectivity') == 'high':
            mat['confidence'] = 'high'
            mat['likely_materials'] = mat.get('possible_materials', [])[:2]
        else:
            mat['confidence'] = 'medium'
    
    return features

def _rgb_to_color_name(rgb):
    """Convert RGB to color name"""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    
    if v < 0.2:
        return "black"
    elif v > 0.8 and s < 0.1:
        return "white"
    elif s < 0.1:
        return "gray"
    elif h < 0.05 or h > 0.95:
        return "red"
    elif h < 0.1:
        return "orange"
    elif h < 0.2:
        return "yellow"
    elif h < 0.4:
        return "green"
    elif h < 0.6:
        return "cyan"
    elif h < 0.7:
        return "blue"
    elif h < 0.9:
        return "purple"
    else:
        return "pink"

def _analyze_material_indicators(image):
    """Analyze material indicators"""
    indicators = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reflectivity (brightness variance)
    brightness_variance = np.var(gray)
    
    if brightness_variance > 5000:
        indicators['reflectivity'] = 'high'
        indicators['possible_materials'] = ['metal', 'glass', 'ceramic', 'polished_wood']
    elif brightness_variance > 2000:
        indicators['reflectivity'] = 'medium'
        indicators['possible_materials'] = ['plastic', 'painted_surface', 'glazed_ceramic', 'enamel']
    else:
        indicators['reflectivity'] = 'low'
        indicators['possible_materials'] = ['wood', 'fabric', 'paper', 'matte_surface', 'rubber']
    
    # Edge patterns (for texture)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    if edge_density > 0.1:
        indicators['texture_level'] = 'high'
        indicators['surface'] = 'textured'
    elif edge_density > 0.05:
        indicators['texture_level'] = 'medium'
        indicators['surface'] = 'slightly_textured'
    else:
        indicators['texture_level'] = 'low'
        indicators['surface'] = 'smooth'
    
    return indicators

def _analyze_condition_indicators(image, object_type):
    """Analyze condition based on visual cues"""
    indicators = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Scratches/damage detection
    edges = cv2.Canny(gray, 30, 100)
    scratch_ratio = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    # Start with perfect score
    condition_score = 10
    
    # Deduct for scratches
    if scratch_ratio > 0.15:
        condition_score -= 4
        indicators['scratches'] = 'heavy'
        indicators['damage_level'] = 'high'
    elif scratch_ratio > 0.08:
        condition_score -= 2
        indicators['scratches'] = 'moderate'
        indicators['damage_level'] = 'medium'
    elif scratch_ratio > 0.03:
        condition_score -= 1
        indicators['scratches'] = 'light'
        indicators['damage_level'] = 'low'
    else:
        indicators['scratches'] = 'none'
        indicators['damage_level'] = 'none'
    
    # Color fading (variance in color)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_variance = np.var(hsv[:, :, 1])
    
    if saturation_variance < 100:
        condition_score -= 2
        indicators['fading'] = 'yes'
        indicators['color_preservation'] = 'poor'
    elif saturation_variance < 200:
        condition_score -= 1
        indicators['fading'] = 'slight'
        indicators['color_preservation'] = 'fair'
    else:
        indicators['fading'] = 'no'
        indicators['color_preservation'] = 'good'
    
    # Ensure score stays within bounds
    condition_score = max(0, min(10, condition_score))
    indicators['condition_score'] = condition_score
    
    # Overall condition assessment
    if condition_score >= 9:
        indicators['overall_condition'] = 'excellent'
    elif condition_score >= 7:
        indicators['overall_condition'] = 'good'
    elif condition_score >= 5:
        indicators['overall_condition'] = 'fair'
    else:
        indicators['overall_condition'] = 'poor'
    
    return indicators

def estimate_condition_from_features(features, object_type):
    """Estimate object condition based on visual features"""
    if not features:
        return "unknown", 0.5
    
    condition_info = features.get('condition_indicators', {})
    condition = condition_info.get('overall_condition', 'unknown')
    score = condition_info.get('condition_score', 5) / 10.0
    
    return condition, score

def generate_detailed_report(object_type, condition, features=None, confidence=0.5):
    """Generate detailed object report"""
    object_lower = object_type.lower()
    
    if object_lower not in ENHANCED_KNOWLEDGE:
        # Generic object
        return {
            'object': object_type,
            'info': f"A {object_type}.",
            'condition': condition,
            'condition_score': 5,
            'estimated_value': 'Unknown',
            'confidence': confidence
        }
    
    knowledge = ENHANCED_KNOWLEDGE[object_lower]
    
    # Estimate value
    base_value = knowledge.get('base_value', 100)
    condition_multiplier = CONDITION_MULTIPLIERS.get(condition, 0.5)
    value = base_value * condition_multiplier
    
    # Apply material multiplier if available
    if features and 'material_indicators' in features:
        materials = features['material_indicators'].get('possible_materials', [])
        for material in materials:
            if material in MATERIAL_VALUE_IMPACT:
                value *= MATERIAL_VALUE_IMPACT[material]
                break
    
    # Apply rarity multiplier
    rarity = knowledge.get('rarity', 'common')
    value *= RARITY_MULTIPLIERS.get(rarity, 1.0)
    
    # Format value
    if value >= 1_000_000:
        estimated_value = f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        estimated_value = f"${value/1_000:.1f}K"
    else:
        estimated_value = f"${value:,.2f}"
    
    report = {
        'object': object_type,
        'info': knowledge['info'],
        'condition': condition,
        'condition_score': 5,
        'estimated_value': estimated_value,
        'value_range': knowledge.get('estimated_value', 'Variable'),
        'brands': knowledge.get('brands', ['Various'])[:3],
        'materials': knowledge.get('materials', ['Unknown']),
        'maintenance': knowledge.get('maintenance', 'Handle with care'),
        'authenticity_tips': knowledge.get('authenticity_indicators', ['Check overall quality'])[:3],
        'rarity': knowledge.get('rarity', 'Unknown'),
        'expert_tip': knowledge.get('tip', ''),
        'confidence': confidence,
        'visual_features': features if features else {}
    }
    
    return report

def comprehensive_object_assessment(object_type, bbox, frame, confidence):
    """Perform comprehensive assessment of an object"""
    # Visual analysis with deep features
    features = analyze_object_deep_features(frame, bbox, object_type)
    
    # Get knowledge base info
    object_lower = object_type.lower()
    if object_lower in ENHANCED_KNOWLEDGE:
        knowledge = ENHANCED_KNOWLEDGE[object_lower]
    else:
        knowledge = {
            "info": f"A {object_type}.",
            "value": "Variable",
            "tip": "No specific information available.",
            "brands": ["Various"],
            "conditions": ["unknown"],
            "materials": ["unknown"],
            "estimated_value": "Unknown",
            "maintenance": "Handle with care",
            "rarity": "Unknown",
            "authenticity_indicators": ["Check overall quality"]
        }
    
    # Determine condition
    condition_info = features.get('condition_indicators', {}) if features else {}
    condition = condition_info.get('overall_condition', 'unknown')
    
    # Generate assessment
    assessment = {
        "identification": {
            "object_type": object_type,
            "confidence": confidence,
            "alternatives": []
        },
        "condition": {
            "rating": condition,
            "score": condition_info.get('condition_score', 5),
            "details": condition_info
        },
        "visual_characteristics": {
            "colors": [features.get('color_name', 'unknown')] if features else ['unknown'],
            "texture": features.get('material_indicators', {}).get('texture_level', 'unknown') if features else 'unknown',
            "size": {
                "relative": features.get('relative_size', 0) if features else 0,
                "aspect_ratio": features.get('aspect_ratio', 0) if features else 0
            }
        },
        "value_assessment": {
            "estimated_value": "Unknown"
        }
    }
    
    # Estimate value
    if object_lower in ENHANCED_KNOWLEDGE:
        base_value = ENHANCED_KNOWLEDGE[object_lower].get('base_value', 100)
        condition_multiplier = CONDITION_MULTIPLIERS.get(condition, 0.5)
        value = base_value * condition_multiplier
        
        if value >= 1_000_000:
            assessment["value_assessment"]["estimated_value"] = f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            assessment["value_assessment"]["estimated_value"] = f"${value/1_000:.1f}K"
        else:
            assessment["value_assessment"]["estimated_value"] = f"${value:,.2f}"
    
    return assessment

def format_report_for_display(report):
    """Format report for on-screen display"""
    lines = []
    
    lines.append(f"📦 OBJECT: {report['object'].upper()}")
    lines.append(f"📊 Condition: {report['condition'].upper()}")
    lines.append(f"💎 Estimated Value: {report['estimated_value']}")
    
    if 'brands' in report and report['brands'] != ["Various"]:
        lines.append(f"🏷️ Common Brands: {', '.join(report['brands'][:3])}")
    
    if 'expert_tip' in report and report['expert_tip']:
        lines.append(f"💡 Tip: {report['expert_tip'][:100]}...")
    
    return lines

def format_comprehensive_assessment(assessment):
    """Format comprehensive assessment for display"""
    lines = []
    
    lines.append("="*50)
    lines.append(f"📊 COMPREHENSIVE ASSESSMENT: {assessment['identification']['object_type'].upper()}")
    lines.append("="*50)
    lines.append(f"🎯 Confidence: {assessment['identification']['confidence']:.2f}")
    lines.append(f"📊 Condition: {assessment['condition']['rating'].upper()} ({assessment['condition']['score']}/10)")
    lines.append(f"💰 Estimated Value: {assessment['value_assessment']['estimated_value']}")
    
    if assessment['visual_characteristics']['colors'] != ['unknown']:
        lines.append(f"🎨 Colors: {', '.join(assessment['visual_characteristics']['colors'])}")
    
    lines.append("="*50)
    
    return lines

# Simple knowledge base interface for backward compatibility
class SimpleKnowledgeBase:
    def __init__(self):
        self.objects = ENHANCED_KNOWLEDGE
    
    def get_object_info(self, object_name):
        return self.objects.get(object_name.lower())
    
    def analyze_object_visual_features(self, frame, bbox, object_type):
        return analyze_object_deep_features(frame, bbox, object_type)
    
    def generate_detailed_report(self, object_type, condition, features=None, confidence=0.5):
        return generate_detailed_report(object_type, condition, features, confidence)
    
    def estimate_object_value(self, object_type, condition='good', features=None):
        object_lower = object_type.lower()
        if object_lower not in self.objects:
            return "$Unknown"
        
        base_value = self.objects[object_lower].get('base_value', 100)
        condition_multiplier = CONDITION_MULTIPLIERS.get(condition, 0.5)
        value = base_value * condition_multiplier
        
        if value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.2f}"

# Create global instance
knowledge_base = SimpleKnowledgeBase()
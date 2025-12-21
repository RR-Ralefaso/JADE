
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
        "authenticity_indicators": ["normal proportions", "natural movement"]
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
        "authenticity_indicators": ["brand logos", "serial numbers", "quality welds"]
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
        "authenticity_indicators": ["VIN number", "original parts", "service records"]
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
        "authenticity_indicators": ["brand tags", "quality stitching", "material feel"]
    },
    
    "handbag": {
        "info": "Women's fashion accessory. Status symbol.",
        "value": "Designer brands (Louis Vuitton, Gucci) thousands.",
        "tip": "Check serial numbers for authenticity.",
        "brands": ["Louis Vuitton", "Gucci", "Chanel", "Prada", "Hermes"],
        "conditions": ["brand_new", "excellent", "good", "fair", "poor"],
        "materials": ["leather", "canvas", "suede", "exotic_skins"],
        "estimated_value": "$100 - $50,000+",
        "maintenance": "Store with stuffing, avoid moisture, condition leather",
        "rarity": "Common to Ultra Rare",
        "authenticity_indicators": ["serial numbers", "quality hardware", "stitching pattern"]
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
        "authenticity_indicators": ["serial numbers", "OS authenticity", "build quality"]
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
        "authenticity_indicators": ["IMEI check", "original box", "software authenticity"]
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
        "authenticity_indicators": ["brand markings", "material quality", "weight"]
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
        "authenticity_indicators": ["brand markings", "construction quality", "material authenticity"]
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
        "authenticity_indicators": ["ISBN", "printing details", "signature verification"]
    },
    
    "clock": {
        "info": "Time-keeping device.",
        "value": "Antique/vintage clocks collectible.",
        "tip": "Check if mechanical works.",
        "brands": ["Rolex", "Omega", "Seiko", "Cartier", "Grandfather clocks"],
        "conditions": ["new", "vintage", "antique", "needs_repair", "working"],
        "materials": ["wood", "metal", "plastic", "glass"],
        "estimated_value": "$10 - $50,000+",
        "maintenance": "Regular winding, professional servicing",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["brand markings", "movement quality", "serial numbers"]
    },
    
    "vase": {
        "info": "Ornamental container.",
        "value": "Antique/designer vases = high value.",
        "tip": "Check for chips/cracks and maker's marks.",
        "brands": ["Royal Copenhagen", "Wedgwood", "Meissen", "Ming dynasty"],
        "conditions": ["perfect", "excellent", "good", "damaged", "antique"],
        "materials": ["porcelain", "ceramic", "glass", "crystal"],
        "estimated_value": "$10 - $100,000+",
        "maintenance": "Handle carefully, dust regularly",
        "rarity": "Common to Ultra Rare",
        "authenticity_indicators": ["maker's marks", "age signs", "material quality"]
    }
}

CONDITION_ASSESSMENT = {
    "excellent": {
        "score": 9,
        "description": "Near perfect, minimal signs of use",
        "value_multiplier": 0.9,
        "visual_indicators": ["minimal wear", "clean", "no damage"]
    },
    "good": {
        "score": 7,
        "description": "Normal wear, fully functional",
        "value_multiplier": 0.7,
        "visual_indicators": ["light wear", "minor scuffs", "fully functional"]
    },
    "fair": {
        "score": 5,
        "description": "Visible wear, needs attention",
        "value_multiplier": 0.5,
        "visual_indicators": ["noticeable wear", "minor damage", "functional"]
    },
    "poor": {
        "score": 3,
        "description": "Significant wear/damage",
        "value_multiplier": 0.3,
        "visual_indicators": ["heavy wear", "damage present", "may need repair"]
    }
}

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

def analyze_object_visual_features(frame, bbox, object_type):
    """
    Analyze visual features of detected object
    """
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = map(int, bbox)
    obj_image = frame[y1:y2, x1:x2]
    
    if obj_image.size == 0:
        return None
    
    features = {}
    
    # Basic color analysis
    hsv = cv2.cvtColor(obj_image, cv2.COLOR_BGR2HSV)
    
    # Color dominance
    colors, counts = np.unique(obj_image.reshape(-1, 3), axis=0, return_counts=True)
    dominant_color = colors[np.argmax(counts)]
    features['dominant_color'] = dominant_color.tolist()
    
    # Edge density (indicates texture/wear)
    gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (obj_image.shape[0] * obj_image.shape[1])
    features['texture_density'] = float(edge_density)
    
    # Brightness/vibrancy
    brightness = np.mean(gray)
    features['brightness'] = float(brightness)
    
    # Color variance (indicates patterns/multiple colors)
    color_variance = np.std(obj_image)
    features['color_variance'] = float(color_variance)
    
    # Sharpness/blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['sharpness'] = float(laplacian_var)
    
    return features

def estimate_condition_from_features(features, object_type):
    """
    Estimate object condition based on visual features
    """
    if not features:
        return "unknown", 0.5
    
    # Different condition indicators for different object types
    condition_rules = {
        "car": {
            "excellent": features['sharpness'] > 100 and features['color_variance'] < 30,
            "good": features['sharpness'] > 50,
            "fair": features['sharpness'] > 20,
            "poor": features['sharpness'] <= 20
        },
        "electronic": {
            "excellent": features['sharpness'] > 150,
            "good": features['sharpness'] > 80,
            "fair": features['sharpness'] > 40,
            "poor": features['sharpness'] <= 40
        },
        "furniture": {
            "excellent": features['texture_density'] < 0.05,
            "good": features['texture_density'] < 0.1,
            "fair": features['texture_density'] < 0.2,
            "poor": features['texture_density'] >= 0.2
        },
        "default": {
            "excellent": features['sharpness'] > 100 and features['texture_density'] < 0.05,
            "good": features['sharpness'] > 50,
            "fair": features['sharpness'] > 20,
            "poor": True
        }
    }
    
    # Determine object category
    object_categories = {
        "car": ["car", "truck", "bus", "motorcycle"],
        "electronic": ["laptop", "cell phone", "tv", "remote", "keyboard", "mouse"],
        "furniture": ["chair", "couch", "bed", "dining table"],
        "default": []
    }
    
    category = "default"
    for cat, items in object_categories.items():
        if object_type in items:
            category = cat
            break
    
    rules = condition_rules.get(category, condition_rules["default"])
    
    # Apply rules
    if rules.get("excellent", False):
        return "excellent", 0.9
    elif rules.get("good", False):
        return "good", 0.7
    elif rules.get("fair", False):
        return "fair", 0.5
    else:
        return "poor", 0.3

def generate_detailed_report(object_type, condition, features=None, confidence=0.5):
    """
    Generate comprehensive report for any detected object
    """
    if object_type not in ENHANCED_KNOWLEDGE:
        # Fallback to basic knowledge or generic report
        return {
            "object": object_type,
            "info": f"A {object_type}. No detailed information available.",
            "condition": condition,
            "condition_score": 5,
            "estimated_value": "Unknown",
            "maintenance": "Handle with care",
            "authenticity_tips": ["Check for obvious signs of quality"],
            "analysis_confidence": confidence
        }
    
    knowledge = ENHANCED_KNOWLEDGE[object_type]
    condition_info = CONDITION_ASSESSMENT.get(condition, CONDITION_ASSESSMENT["good"])
    
    # Calculate value estimate
    base_values = {
        "car": 20000,
        "laptop": 800,
        "cell phone": 500,
        "bicycle": 300,
        "handbag": 200,
        "backpack": 50,
        "chair": 150,
        "bottle": 25,
        "book": 20,
        "clock": 100,
        "vase": 75,
        "default": 100
    }
    
    base_value = base_values.get(object_type, base_values["default"])
    adjusted_value = base_value * condition_info["value_multiplier"]
    
    # Material impact
    if "materials" in knowledge and knowledge["materials"][0] in MATERIAL_VALUE_IMPACT:
        material_multiplier = MATERIAL_VALUE_IMPACT[knowledge["materials"][0]]
        adjusted_value *= material_multiplier
    
    report = {
        "object": object_type,
        "info": knowledge["info"],
        "condition": condition,
        "condition_score": condition_info["score"],
        "condition_description": condition_info["description"],
        "estimated_value": f"${adjusted_value:,.2f}",
        "value_range": knowledge.get("estimated_value", "Variable"),
        "brands": knowledge.get("brands", ["Various"]),
        "materials": knowledge.get("materials", ["Unknown"]),
        "maintenance": knowledge.get("maintenance", "Handle with care"),
        "authenticity_tips": knowledge.get("authenticity_indicators", ["Check overall quality"]),
        "rarity": knowledge.get("rarity", "Unknown"),
        "expert_tip": knowledge.get("tip", ""),
        "analysis_confidence": confidence,
        "visual_features": features if features else {}
    }
    
    return report

def format_report_for_display(report):
    """
    Format report for on-screen display
    """
    lines = []
    
    lines.append(f"📦 OBJECT: {report['object'].upper()}")
    lines.append(f"📊 Condition: {report['condition'].upper()} ({report['condition_score']}/10)")
    lines.append(f"💎 Estimated Value: {report['estimated_value']}")
    lines.append(f"🏷️ Common Brands: {', '.join(report['brands'][:3])}")
    lines.append(f"🔍 Info: {report['info']}")
    lines.append(f"💡 Tip: {report['expert_tip']}")
    lines.append(f"🛠️ Maintenance: {report['maintenance']}")
    
    return lines

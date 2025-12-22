import numpy as np
import colorsys
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops

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
    
    "motorcycle": {
        "info": "Two or three-wheeled motor vehicle.",
        "value": "Check engine size and brand for valuation.",
        "tip": "Collectible models appreciate in value.",
        "brands": ["Harley-Davidson", "Honda", "Yamaha", "Kawasaki", "Ducati"],
        "conditions": ["new", "used", "vintage", "custom", "restored"],
        "materials": ["steel", "aluminum", "chrome", "leather"],
        "estimated_value": "$1,000 - $50,000+",
        "maintenance": "Chain/belt maintenance, oil changes, tire pressure",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["VIN", "engine numbers", "original parts"]
    },
    
    "airplane": {
        "info": "Fixed-wing aircraft.",
        "value": "Extremely high value, depends on type and condition.",
        "tip": "Private jets hold value better than commercial aircraft.",
        "brands": ["Boeing", "Airbus", "Cessna", "Gulfstream", "Embraer"],
        "conditions": ["airworthy", "needs_maintenance", "restoration_project"],
        "materials": ["aluminum", "composite", "titanium"],
        "estimated_value": "$100,000 - $100,000,000+",
        "maintenance": "Regular inspections, engine overhauls, avionics updates",
        "rarity": "Rare",
        "authenticity_indicators": ["registration", "logbooks", "manufacturer plates"]
    },
    
    "bus": {
        "info": "Large passenger vehicle.",
        "value": "Depreciates quickly unless vintage/collectible.",
        "tip": "School buses can be converted to RVs.",
        "brands": ["Blue Bird", "Thomas", "Mercedes", "Volvo"],
        "conditions": ["new", "used", "high_mileage", "vintage"],
        "materials": ["steel", "fiberglass", "aluminum"],
        "estimated_value": "$20,000 - $500,000",
        "maintenance": "Engine service, transmission, brake systems",
        "rarity": "Common",
        "authenticity_indicators": ["VIN", "title", "inspection records"]
    },
    
    "train": {
        "info": "Rail transport vehicle.",
        "value": "Very high for working locomotives.",
        "tip": "Vintage trains are highly collectible.",
        "brands": ["General Electric", "EMD", "Siemens", "Bombardier"],
        "conditions": ["operational", "restored", "needs_restoration", "vintage"],
        "materials": ["steel", "cast_iron", "brass"],
        "estimated_value": "$50,000 - $5,000,000+",
        "maintenance": "Track maintenance, engine service, brake systems",
        "rarity": "Rare",
        "authenticity_indicators": ["serial numbers", "builder plates", "documentation"]
    },
    
    "truck": {
        "info": "Commercial or personal hauling vehicle.",
        "value": "Diesel trucks hold value better than gasoline.",
        "tip": "Low mileage work trucks are premium.",
        "brands": ["Ford", "Chevrolet", "Ram", "Toyota", "Isuzu"],
        "conditions": ["new", "used", "work_ready", "needs_repair"],
        "materials": ["steel", "aluminum", "fiberglass"],
        "estimated_value": "$10,000 - $150,000",
        "maintenance": "Engine maintenance, transmission, suspension",
        "rarity": "Common",
        "authenticity_indicators": ["VIN", "title status", "service records"]
    },
    
    "boat": {
        "info": "Watercraft for transportation or recreation.",
        "value": "Fiberglass boats depreciate, wood boats can appreciate.",
        "tip": "Check hull condition and engine hours.",
        "brands": ["Sea Ray", "Bayliner", "Yamaha", "Boston Whaler", "Mastercraft"],
        "conditions": ["new", "used", "excellent", "project_boat"],
        "materials": ["fiberglass", "aluminum", "wood", "steel"],
        "estimated_value": "$5,000 - $5,000,000+",
        "maintenance": "Hull cleaning, engine winterization, electronics",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["HIN", "title", "registration"]
    },
    
    "traffic light": {
        "info": "Signal for controlling traffic.",
        "value": "Vintage signals are collectible.",
        "tip": "Working antique signals command premium prices.",
        "brands": ["Econolite", "McCain", "Siemens", "Crouse-Hinds"],
        "conditions": ["working", "vintage", "antique", "needs_repair"],
        "materials": ["steel", "aluminum", "glass", "plastic"],
        "estimated_value": "$100 - $5,000",
        "maintenance": "Bulb replacement, lens cleaning, electrical checks",
        "rarity": "Common",
        "authenticity_indicators": ["manufacturer marks", "age patina", "original components"]
    },
    
    "fire hydrant": {
        "info": "Water supply connection for firefighting.",
        "value": "Antique cast iron hydrants are collectible.",
        "tip": "Color indicates water flow capacity.",
        "brands": ["Mueller", "Kennedy", "American-Darling"],
        "conditions": ["working", "vintage", "antique", "decorative"],
        "materials": ["cast_iron", "bronze", "ductile_iron"],
        "estimated_value": "$50 - $2,000",
        "maintenance": "Regular flushing, painting, valve checks",
        "rarity": "Common",
        "authenticity_indicators": ["foundry marks", "patina", "style period"]
    },
    
    "stop sign": {
        "info": "Regulatory traffic sign.",
        "value": "Vintage signs with original paint are collectible.",
        "tip": "1950s-60s signs with porcelain enamel are most valuable.",
        "brands": ["Various municipalities"],
        "conditions": ["new", "used", "vintage", "rusty", "restored"],
        "materials": ["aluminum", "steel", "porcelain_enamel", "reflective_sheeting"],
        "estimated_value": "$20 - $500",
        "maintenance": "Clean reflective surface, repaint as needed",
        "rarity": "Common",
        "authenticity_indicators": ["age patina", "manufacturer stamps", "paint type"]
    },
    
    "parking meter": {
        "info": "Coin-operated timing device for parking.",
        "value": "Vintage mechanical meters are collectible.",
        "tip": "Complete with keys and working mechanism increases value.",
        "brands": ["Duncan", "M.H. Rhodes", "Grob", "POM"],
        "conditions": ["working", "vintage", "antique", "needs_repair"],
        "materials": ["cast_iron", "steel", "brass", "chrome"],
        "estimated_value": "$50 - $1,500",
        "maintenance": "Mechanism cleaning, coin slot maintenance",
        "rarity": "Uncommon",
        "authenticity_indicators": ["manufacturer plates", "original paint", "working mechanism"]
    },
    
    "bench": {
        "info": "Long seat for multiple people.",
        "value": "Designer/antique benches command high prices.",
        "tip": "Check for maker's marks on cast iron benches.",
        "brands": ["Walt Disney", "J.L. Mott", "Woodard", "Troy"],
        "conditions": ["new", "used", "antique", "garden", "needs_restoration"],
        "materials": ["wood", "cast_iron", "wrought_iron", "stone", "concrete"],
        "estimated_value": "$100 - $10,000",
        "maintenance": "Regular cleaning, rust prevention, wood treatment",
        "rarity": "Common",
        "authenticity_indicators": ["foundry marks", "design style", "construction methods"]
    },
    
    "bird": {
        "info": "Feathered animal, often kept as pet.",
        "value": "Rare species and trained birds are valuable.",
        "tip": "Documentation and health records increase value.",
        "brands": ["N/A"],
        "conditions": ["healthy", "trained", "breeding", "show_quality"],
        "materials": ["feathers", "biological"],
        "estimated_value": "$20 - $20,000+",
        "maintenance": "Proper diet, cage cleaning, veterinary care",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["species documentation", "breeding papers", "health records"]
    },
    
    "cat": {
        "info": "Domestic feline pet.",
        "value": "Pedigree cats with papers are valuable.",
        "tip": "Show quality cats with championships command highest prices.",
        "brands": ["N/A"],
        "conditions": ["healthy", "pedigree", "show_quality", "breeding"],
        "materials": ["fur", "biological"],
        "estimated_value": "$0 - $10,000+",
        "maintenance": "Veterinary care, proper nutrition, grooming",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["pedigree papers", "registration", "health certificates"]
    },
    
    "dog": {
        "info": "Domestic canine pet.",
        "value": "Working breeds and show dogs are most valuable.",
        "tip": "Trained service/therapy dogs have added value.",
        "brands": ["N/A"],
        "conditions": ["healthy", "trained", "pedigree", "show_quality", "working"],
        "materials": ["fur", "biological"],
        "estimated_value": "$0 - $20,000+",
        "maintenance": "Veterinary care, training, exercise, grooming",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["pedigree papers", "training certificates", "health records"]
    },
    
    "horse": {
        "info": "Large domesticated mammal.",
        "value": "Bloodlines and training determine value.",
        "tip": "Competition records significantly increase value.",
        "brands": ["N/A"],
        "conditions": ["healthy", "trained", "competition", "breeding"],
        "materials": ["biological"],
        "estimated_value": "$500 - $500,000+",
        "maintenance": "Stabling, feeding, veterinary care, training",
        "rarity": "Uncommon",
        "authenticity_indicators": ["registration papers", "bloodline documentation", "competition records"]
    },
    
    "sheep": {
        "info": "Domesticated ruminant animal.",
        "value": "Specialty breeds for wool or meat are valuable.",
        "tip": "Registered breeding stock commands premium.",
        "brands": ["N/A"],
        "conditions": ["healthy", "breeding", "wool_production"],
        "materials": ["wool", "biological"],
        "estimated_value": "$100 - $5,000",
        "maintenance": "Grazing management, shearing, veterinary care",
        "rarity": "Common",
        "authenticity_indicators": ["breed registration", "health records", "wool quality"]
    },
    
    "cow": {
        "info": "Domesticated bovine animal.",
        "value": "Dairy and beef breeds have different valuations.",
        "tip": "Show champions and proven breeding stock are most valuable.",
        "brands": ["N/A"],
        "conditions": ["healthy", "dairy", "beef", "breeding", "show"],
        "materials": ["biological"],
        "estimated_value": "$500 - $50,000+",
        "maintenance": "Feeding, milking (if dairy), veterinary care",
        "rarity": "Common",
        "authenticity_indicators": ["registration papers", "milk records", "breeding history"]
    },
    
    "elephant": {
        "info": "Large terrestrial mammal.",
        "value": "Extremely high for trained/working elephants.",
        "tip": "Legal documentation is crucial due to protection laws.",
        "brands": ["N/A"],
        "conditions": ["healthy", "trained", "working", "sanctuary"],
        "materials": ["biological"],
        "estimated_value": "$30,000 - $100,000+",
        "maintenance": "Specialized care, large habitat, veterinary expertise",
        "rarity": "Rare",
        "authenticity_indicators": ["legal documentation", "health records", "training certificates"]
    },
    
    "bear": {
        "info": "Large carnivorous mammal.",
        "value": "Highly regulated, mostly in zoos/sanctuaries.",
        "tip": "Never keep as private pet - illegal in most places.",
        "brands": ["N/A"],
        "conditions": ["wild", "zoo", "sanctuary"],
        "materials": ["fur", "biological"],
        "estimated_value": "Priceless/Not for sale",
        "maintenance": "Large natural habitat, specialized diet, expert care",
        "rarity": "Protected",
        "authenticity_indicators": ["zoo/sanctuary documentation", "legal permits"]
    },
    
    "zebra": {
        "info": "African equid with black and white stripes.",
        "value": "Zoo/sanctuary animals, not private pets.",
        "tip": "Each zebra has unique stripe pattern like fingerprints.",
        "brands": ["N/A"],
        "conditions": ["wild", "zoo", "conservation"],
        "materials": ["biological"],
        "estimated_value": "$5,000 - $30,000 (zoo transfer)",
        "maintenance": "Large space, specialized diet, herd socialization",
        "rarity": "Uncommon",
        "authenticity_indicators": ["zoo documentation", "conservation status"]
    },
    
    "giraffe": {
        "info": "Tallest living terrestrial animal.",
        "value": "Zoo animals with conservation value.",
        "tip": "Require specialized facilities due to height.",
        "brands": ["N/A"],
        "conditions": ["wild", "zoo", "conservation"],
        "materials": ["biological"],
        "estimated_value": "$25,000 - $100,000+ (zoo transfer)",
        "maintenance": "Tall enclosures, specialized feeding stations, veterinary care",
        "rarity": "Uncommon",
        "authenticity_indicators": ["zoo breeding programs", "conservation documentation"]
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
    
    "umbrella": {
        "info": "Canopy for rain or sun protection.",
        "value": "Designer and antique umbrellas are collectible.",
        "tip": "Check mechanism and fabric condition.",
        "brands": ["Fulton", "Davek", "Blunt", "Senz", "Burberry"],
        "conditions": ["new", "used", "vintage", "broken"],
        "materials": ["nylon", "polyester", "pongee", "wood", "metal"],
        "estimated_value": "$10 - $500",
        "maintenance": "Dry before storing, clean fabric, lubricate mechanism",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "quality of mechanism", "material quality"]
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
    
    "tie": {
        "info": "Neckwear for formal attire.",
        "value": "Designer and vintage ties are collectible.",
        "tip": "Silk ties from 1920s-1950s are most valuable.",
        "brands": ["Hermes", "Gucci", "Brooks Brothers", "Charvet", "Drakes"],
        "conditions": ["new", "vintage", "antique", "used", "stained"],
        "materials": ["silk", "wool", "polyester", "linen"],
        "estimated_value": "$10 - $500",
        "maintenance": "Dry clean only, store rolled, avoid wrinkles",
        "rarity": "Common",
        "authenticity_indicators": ["brand labels", "hand rolling", "fabric quality"]
    },
    
    "suitcase": {
        "info": "Luggage for travel.",
        "value": "Vintage leather suitcases are collectible.",
        "tip": "Check for original labels and hardware.",
        "brands": ["Louis Vuitton", "Gucci", "Hartmann", "Globe-Trotter", "Rimowa"],
        "conditions": ["new", "vintage", "antique", "used", "restored"],
        "materials": ["leather", "canvas", "hard_shell", "aluminum"],
        "estimated_value": "$50 - $5,000",
        "maintenance": "Clean exterior, condition leather, repair hardware",
        "rarity": "Common",
        "authenticity_indicators": ["brand stamps", "age patina", "construction quality"]
    },
    
    "frisbee": {
        "info": "Flying disc for sport or recreation.",
        "value": "First edition Wham-O discs are collectible.",
        "tip": "Check for patent numbers and early logos.",
        "brands": ["Wham-O", "Discraft", "Innova", "MVP"],
        "conditions": ["new", "used", "vintage", "collectible"],
        "materials": ["plastic", "rubber"],
        "estimated_value": "$5 - $500",
        "maintenance": "Clean with mild soap, avoid extreme heat",
        "rarity": "Common",
        "authenticity_indicators": ["manufacturer marks", "patent numbers", "logo variations"]
    },
    
    "skis": {
        "info": "Snow sports equipment.",
        "value": "Vintage wood skis and modern high-end skis are valuable.",
        "tip": "Check for cracks in the base and edge condition.",
        "brands": ["Rossignol", "K2", "Atomic", "Volkl", "Head"],
        "conditions": ["new", "used", "vintage", "needs_tuning"],
        "materials": ["wood", "fiberglass", "carbon", "metal"],
        "estimated_value": "$50 - $1,500",
        "maintenance": "Regular waxing, edge sharpening, storage in dry place",
        "rarity": "Common",
        "authenticity_indicators": ["brand graphics", "construction quality", "bindings"]
    },
    
    "snowboard": {
        "info": "Single board for snow riding.",
        "value": "Limited edition and pro models hold value.",
        "tip": "Check for base damage and edge separation.",
        "brands": ["Burton", "K2", "Ride", "Lib Tech", "Never Summer"],
        "conditions": ["new", "used", "vintage", "needs_repair"],
        "materials": ["wood", "fiberglass", "carbon", "plastic"],
        "estimated_value": "$100 - $800",
        "maintenance": "Wax regularly, sharpen edges, repair base damage",
        "rarity": "Common",
        "authenticity_indicators": ["brand graphics", "serial numbers", "construction"]
    },
    
    "sports ball": {
        "info": "Ball for various sports.",
        "value": "Game-used and autographed balls are most valuable.",
        "tip": "Check for signatures and authentication.",
        "brands": ["Wilson", "Spalding", "Nike", "Adidas", "Rawlings"],
        "conditions": ["new", "used", "game_used", "autographed"],
        "materials": ["leather", "rubber", "synthetic"],
        "estimated_value": "$10 - $10,000+",
        "maintenance": "Clean with appropriate products, proper inflation",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["authentication holograms", "league markings", "signature verification"]
    },
    
    "kite": {
        "info": "Tethered flying object.",
        "value": "Antique and handmade kites are collectible.",
        "tip": "Check for original paint and intact framework.",
        "brands": ["Premier", "Prism", "Into the Wind", "HQ"],
        "conditions": ["new", "vintage", "antique", "handmade", "needs_repair"],
        "materials": ["silk", "paper", "nylon", "bamboo", "carbon"],
        "estimated_value": "$20 - $1,000",
        "maintenance": "Clean fabric, repair tears, store dry",
        "rarity": "Common",
        "authenticity_indicators": ["hand craftsmanship", "materials", "age indicators"]
    },
    
    "baseball bat": {
        "info": "Club for hitting baseballs.",
        "value": "Game-used and autographed bats command premium.",
        "tip": "Check for pine tar residue and ball marks.",
        "brands": ["Louisville Slugger", "Marucci", "Old Hickory", "Rawlings"],
        "conditions": ["new", "used", "game_used", "autographed", "vintage"],
        "materials": ["ash", "maple", "birch", "aluminum"],
        "estimated_value": "$30 - $10,000+",
        "maintenance": "Clean with mild soap, condition wood, store properly",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["authentication", "game use marks", "signature verification"]
    },
    
    "baseball glove": {
        "info": "Leather mitt for catching baseballs.",
        "value": "Vintage gloves and pro models are collectible.",
        "tip": "Look for manufacturer stamps and player endorsements.",
        "brands": ["Wilson", "Rawlings", "Mizuno", "Nike", "Spalding"],
        "conditions": ["new", "broken_in", "vintage", "game_used", "autographed"],
        "materials": ["leather", "latex", "rubber"],
        "estimated_value": "$50 - $5,000",
        "maintenance": "Condition leather, reshape when storing",
        "rarity": "Common",
        "authenticity_indicators": ["manufacturer stamps", "player endorsements", "age patina"]
    },
    
    "skateboard": {
        "info": "Board with wheels for riding.",
        "value": "Vintage boards and limited editions are collectible.",
        "tip": "Check for original graphics and truck brands.",
        "brands": ["Santa Cruz", "Powell Peralta", "Birdhouse", "Element", "Plan B"],
        "conditions": ["new", "used", "vintage", "collectible", "ridden"],
        "materials": ["maple", "bamboo", "carbon", "plastic"],
        "estimated_value": "$50 - $2,000",
        "maintenance": "Clean bearings, replace grip tape, tighten trucks",
        "rarity": "Common",
        "authenticity_indicators": ["graphic condition", "truck brands", "wheel brands"]
    },
    
    "surfboard": {
        "info": "Board for riding ocean waves.",
        "value": "Vintage longboards and shaped by famous shapers are valuable.",
        "tip": "Check for pressure dings and yellowing.",
        "brands": ["Channel Islands", "Lost", "Firewire", "JS", "Hayden"],
        "conditions": ["new", "used", "vintage", "custom", "needs_repair"],
        "materials": ["fiberglass", "polyurethane", "epoxy", "balsa"],
        "estimated_value": "$200 - $5,000",
        "maintenance": "Repair dings, clean wax, store properly",
        "rarity": "Common",
        "authenticity_indicators": ["shaper signature", "glassing quality", "age indicators"]
    },
    
    "tennis racket": {
        "info": "Implement for striking tennis ball.",
        "value": "Vintage wood rackets and pro models are collectible.",
        "tip": "Check for warping and string condition.",
        "brands": ["Wilson", "Head", "Babolat", "Yonex", "Prince"],
        "conditions": ["new", "used", "vintage", "pro_model", "needs_stringing"],
        "materials": ["wood", "graphite", "aluminum", "composite"],
        "estimated_value": "$30 - $1,000",
        "maintenance": "Restring regularly, store in press, clean grip",
        "rarity": "Common",
        "authenticity_indicators": ["model markings", "player endorsements", "construction"]
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
    
    "wine glass": {
        "info": "Stemware for drinking wine.",
        "value": "Crystal and antique glasses are collectible.",
        "tip": "Check for maker's marks and lead content.",
        "brands": ["Riedel", "Baccarat", "Waterford", "Spiegelau", "Zalto"],
        "conditions": ["new", "used", "antique", "crystal", "etched"],
        "materials": ["crystal", "glass", "lead_crystal"],
        "estimated_value": "$10 - $500",
        "maintenance": "Hand wash, avoid dishwasher, careful storage",
        "rarity": "Common",
        "authenticity_indicators": ["maker's marks", "clarity", "ring tone"]
    },
    
    "cup": {
        "info": "Drinking vessel.",
        "value": "Antique china and collectible mugs are valuable.",
        "tip": "Check for maker's marks and condition.",
        "brands": ["Wedgwood", "Royal Doulton", "Spode", "Lenox", "Starbucks"],
        "conditions": ["new", "used", "antique", "collectible", "chipped"],
        "materials": ["porcelain", "ceramic", "stoneware", "glass", "plastic"],
        "estimated_value": "$5 - $500",
        "maintenance": "Clean appropriately, avoid thermal shock",
        "rarity": "Common",
        "authenticity_indicators": ["maker's marks", "backstamps", "quality"]
    },
    
    "fork": {
        "info": "Eating utensil with tines.",
        "value": "Sterling silver and antique patterns are valuable.",
        "tip": "Check for hallmarks and pattern completeness.",
        "brands": ["Tiffany", "Gorham", "Reed & Barton", "Wallace", "Oneida"],
        "conditions": ["new", "used", "antique", "sterling", "plated"],
        "materials": ["sterling_silver", "silver_plate", "stainless_steel"],
        "estimated_value": "$5 - $200",
        "maintenance": "Polish silver, hand wash antique pieces",
        "rarity": "Common",
        "authenticity_indicators": ["hallmarks", "pattern marks", "weight"]
    },
    
    "knife": {
        "info": "Cutting utensil or tool.",
        "value": "Custom and antique knives are collectible.",
        "tip": "Check blade steel and maker's marks.",
        "brands": ["Benchmade", "Spyderco", "Chris Reeve", "Microtech", "Victorinox"],
        "conditions": ["new", "used", "custom", "antique", "needs_sharpening"],
        "materials": ["steel", "titanium", "carbon_fiber", "wood", "bone"],
        "estimated_value": "$20 - $5,000+",
        "maintenance": "Sharpening, oiling pivots, cleaning",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["maker's marks", "steel markings", "construction quality"]
    },
    
    "spoon": {
        "info": "Eating utensil with bowl.",
        "value": "Sterling silver and souvenir spoons are collectible.",
        "tip": "Check for hallmarks and souvenir markings.",
        "brands": ["Tiffany", "Gorham", "Reed & Barton", "Wallace", "Oneida"],
        "conditions": ["new", "used", "antique", "sterling", "souvenir"],
        "materials": ["sterling_silver", "silver_plate", "stainless_steel"],
        "estimated_value": "$5 - $200",
        "maintenance": "Polish silver, hand wash antique pieces",
        "rarity": "Common",
        "authenticity_indicators": ["hallmarks", "souvenir markings", "weight"]
    },
    
    "bowl": {
        "info": "Round dish for food.",
        "value": "Antique pottery and art pottery are valuable.",
        "tip": "Check for maker's marks and glaze quality.",
        "brands": ["Wedgwood", "Royal Worcester", "Meissen", "Roseville", "Weller"],
        "conditions": ["new", "used", "antique", "art_pottery", "chipped"],
        "materials": ["porcelain", "ceramic", "stoneware", "wood", "glass"],
        "estimated_value": "$10 - $5,000",
        "maintenance": "Clean appropriately, avoid thermal shock",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["maker's marks", "glaze quality", "age signs"]
    },
    
    "banana": {
        "info": "Tropical fruit.",
        "value": "Minimal, mostly commodity pricing.",
        "tip": "Organic and specialty varieties command premium.",
        "brands": ["Chiquita", "Dole", "Del Monte"],
        "conditions": ["green", "ripe", "overripe", "organic"],
        "materials": ["fruit", "biological"],
        "estimated_value": "$0.10 - $1.00",
        "maintenance": "Store at room temperature, avoid refrigeration",
        "rarity": "Common",
        "authenticity_indicators": ["brand stickers", "variety", "origin"]
    },
    
    "apple": {
        "info": "Pome fruit.",
        "value": "Heirloom varieties and organic command premium.",
        "tip": "Check for variety and origin.",
        "brands": ["Washington", "New Zealand", "Local orchards"],
        "conditions": ["fresh", "ripe", "organic", "heirloom"],
        "materials": ["fruit", "biological"],
        "estimated_value": "$0.25 - $2.00",
        "maintenance": "Refrigerate for longer storage",
        "rarity": "Common",
        "authenticity_indicators": ["variety identification", "origin labels"]
    },
    
    "sandwich": {
        "info": "Food item with fillings between bread.",
        "value": "Gourmet and artisanal sandwiches command premium.",
        "tip": "Freshness and ingredient quality determine value.",
        "brands": ["Local delis", "Specialty shops"],
        "conditions": ["fresh", "made_to_order", "packaged", "day_old"],
        "materials": ["food", "biological"],
        "estimated_value": "$5 - $25",
        "maintenance": "Eat fresh, refrigerate if not consuming immediately",
        "rarity": "Common",
        "authenticity_indicators": ["ingredient quality", "freshness", "preparation"]
    },
    
    "orange": {
        "info": "Citrus fruit.",
        "value": "Organic and blood oranges command premium.",
        "tip": "Check for variety and juiciness.",
        "brands": ["Florida", "California", "Spain"],
        "conditions": ["fresh", "ripe", "juicy", "organic"],
        "materials": ["fruit", "biological"],
        "estimated_value": "$0.30 - $1.50",
        "maintenance": "Store at cool temperature",
        "rarity": "Common",
        "authenticity_indicators": ["variety", "origin", "freshness"]
    },
    
    "broccoli": {
        "info": "Edible green plant.",
        "value": "Organic and local command premium.",
        "tip": "Check for tight florets and vibrant color.",
        "brands": ["Local farms", "Organic brands"],
        "conditions": ["fresh", "organic", "local", "wilting"],
        "materials": ["vegetable", "biological"],
        "estimated_value": "$1 - $5",
        "maintenance": "Refrigerate, consume within few days",
        "rarity": "Common",
        "authenticity_indicators": ["freshness", "color vibrancy", "origin"]
    },
    
    "carrot": {
        "info": "Root vegetable.",
        "value": "Organic and heirloom varieties command premium.",
        "tip": "Check for crispness and color.",
        "brands": ["Local farms", "Organic brands"],
        "conditions": ["fresh", "organic", "heirloom", "wilting"],
        "materials": ["vegetable", "biological"],
        "estimated_value": "$0.50 - $3.00",
        "maintenance": "Refrigerate, remove greens for longer storage",
        "rarity": "Common",
        "authenticity_indicators": ["freshness", "color", "origin"]
    },
    
    "hot dog": {
        "info": "Cooked sausage in bun.",
        "value": "Gourmet and specialty dogs command premium.",
        "tip": "Check for quality ingredients and preparation.",
        "brands": ["Nathan's", "Hebrew National", "Vienna Beef", "Local butchers"],
        "conditions": ["fresh", "cooked", "gourmet", "street_food"],
        "materials": ["food", "biological"],
        "estimated_value": "$2 - $15",
        "maintenance": "Eat immediately when hot",
        "rarity": "Common",
        "authenticity_indicators": ["ingredient quality", "preparation", "brand"]
    },
    
    "pizza": {
        "info": "Baked dish with toppings.",
        "value": "Artisanal and gourmet pizzas command premium.",
        "tip": "Check for quality ingredients and proper baking.",
        "brands": ["Local pizzerias", "National chains", "Artisanal"],
        "conditions": ["fresh", "hot", "leftover", "frozen"],
        "materials": ["food", "biological"],
        "estimated_value": "$5 - $50",
        "maintenance": "Eat fresh or refrigerate leftovers",
        "rarity": "Common",
        "authenticity_indicators": ["ingredient quality", "crust texture", "preparation"]
    },
    
    "donut": {
        "info": "Sweet fried dough.",
        "value": "Specialty and gourmet donuts command premium.",
        "tip": "Freshness is crucial for quality.",
        "brands": ["Krispy Kreme", "Dunkin'", "Local bakeries", "Voodoo"],
        "conditions": ["fresh", "day_old", "gourmet", "specialty"],
        "materials": ["food", "biological"],
        "estimated_value": "$1 - $6",
        "maintenance": "Best consumed day of making",
        "rarity": "Common",
        "authenticity_indicators": ["freshness", "quality of ingredients", "preparation"]
    },
    
    "cake": {
        "info": "Baked dessert.",
        "value": "Custom and wedding cakes command premium.",
        "tip": "Check for decoration quality and freshness.",
        "brands": ["Local bakeries", "Specialty shops"],
        "conditions": ["fresh", "custom", "wedding", "leftover"],
        "materials": ["food", "biological"],
        "estimated_value": "$10 - $500+",
        "maintenance": "Refrigerate if cream-based, consume within days",
        "rarity": "Common",
        "authenticity_indicators": ["decoration quality", "freshness", "ingredients"]
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
    
    "couch": {
        "info": "Upholstered furniture for seating multiple people.",
        "value": "Designer and high-quality brands hold value.",
        "tip": "Check frame construction and cushion condition.",
        "brands": ["Restoration Hardware", "Pottery Barn", "Ethan Allen", "Joybird", "Article"],
        "conditions": ["new", "excellent", "good", "fair", "needs_reupholstery"],
        "materials": ["wood", "fabric", "leather", "foam", "down"],
        "estimated_value": "$200 - $10,000+",
        "maintenance": "Vacuum regularly, rotate cushions, condition leather",
        "rarity": "Common",
        "authenticity_indicators": ["construction quality", "material tags", "brand labels"]
    },
    
    "potted plant": {
        "info": "Plant grown in container.",
        "value": "Rare species and mature specimens are valuable.",
        "tip": "Check for healthy roots and proper care.",
        "brands": ["Local nurseries", "Specialty growers"],
        "conditions": ["healthy", "mature", "rare", "needs_care", "dying"],
        "materials": ["biological", "soil", "clay", "ceramic"],
        "estimated_value": "$5 - $5,000+",
        "maintenance": "Proper watering, sunlight, fertilization",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["species identification", "health indicators", "age"]
    },
    
    "bed": {
        "info": "Furniture for sleeping.",
        "value": "Designer and antique beds are valuable.",
        "tip": "Check frame construction and mattress compatibility.",
        "brands": ["Tempur-Pedic", "Sleep Number", "IKEA", "Restoration Hardware", "Antique"],
        "conditions": ["new", "used", "antique", "needs_repair"],
        "materials": ["wood", "metal", "upholstered", "leather"],
        "estimated_value": "$100 - $20,000+",
        "maintenance": "Tighten joints, clean upholstery, rotate mattress",
        "rarity": "Common",
        "authenticity_indicators": ["construction quality", "material tags", "brand markings"]
    },
    
    "dining table": {
        "info": "Table for eating meals.",
        "value": "Solid wood and antique tables are valuable.",
        "tip": "Check for solid construction and finish condition.",
        "brands": ["Restoration Hardware", "Pottery Barn", "Ethan Allen", "IKEA", "Antique"],
        "conditions": ["new", "used", "antique", "needs_refinishing"],
        "materials": ["wood", "glass", "metal", "marble"],
        "estimated_value": "$100 - $15,000+",
        "maintenance": "Clean regularly, protect surface, refinish as needed",
        "rarity": "Common",
        "authenticity_indicators": ["construction quality", "wood type", "finish quality"]
    },
    
    "toilet": {
        "info": "Plumbing fixture for sanitation.",
        "value": "Antique and high-efficiency models have value.",
        "tip": "Check for cracks and flushing mechanism.",
        "brands": ["Kohler", "American Standard", "Toto", "Mansfield", "Antique"],
        "conditions": ["new", "used", "antique", "needs_repair"],
        "materials": ["porcelain", "vitreous_china", "plastic"],
        "estimated_value": "$50 - $5,000",
        "maintenance": "Regular cleaning, check seals, replace parts as needed",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "age indicators", "construction quality"]
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
        "authenticity_indicators": ["model numbers", "brand logos", "functionality"]
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
    
    "mouse": {
        "info": "Computer pointing device.",
        "value": "Limited edition and gaming mice hold value.",
        "tip": "Check sensor functionality and button response.",
        "brands": ["Logitech", "Razer", "SteelSeries", "Microsoft", "Apple"],
        "conditions": ["new", "used", "gaming", "vintage", "broken"],
        "materials": ["plastic", "rubber", "metal"],
        "estimated_value": "$10 - $200",
        "maintenance": "Clean sensor, check buttons, update drivers",
        "rarity": "Common",
        "authenticity_indicators": ["model numbers", "brand logos", "functionality"]
    },
    
    "remote": {
        "info": "Device for controlling electronics remotely.",
        "value": "Universal and vintage remotes are collectible.",
        "tip": "Check button functionality and battery compartment.",
        "brands": ["Logitech Harmony", "Universal", "Original equipment"],
        "conditions": ["new", "used", "universal", "vintage", "broken"],
        "materials": ["plastic", "rubber", "electronics"],
        "estimated_value": "$5 - $200",
        "maintenance": "Clean buttons, replace batteries, store properly",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "model compatibility", "functionality"]
    },
    
    "keyboard": {
        "info": "Input device for computers.",
        "value": "Mechanical and vintage keyboards are collectible.",
        "tip": "Check switch type and keycap condition.",
        "brands": ["Corsair", "Razer", "Logitech", "IBM", "Custom built"],
        "conditions": ["new", "used", "mechanical", "vintage", "custom"],
        "materials": ["plastic", "metal", "PBT", "ABS"],
        "estimated_value": "$20 - $500",
        "maintenance": "Clean regularly, lubricate switches, replace keycaps",
        "rarity": "Common",
        "authenticity_indicators": ["switch type", "build quality", "keycap material"]
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
    
    "microwave": {
        "info": "Appliance for heating food.",
        "value": "Commercial and vintage models have value.",
        "tip": "Check heating functionality and door seal.",
        "brands": ["Panasonic", "Samsung", "LG", "Whirlpool", "GE"],
        "conditions": ["new", "used", "commercial", "vintage", "broken"],
        "materials": ["stainless_steel", "plastic", "glass"],
        "estimated_value": "$50 - $1,000",
        "maintenance": "Clean interior, check door seal, replace filter",
        "rarity": "Common",
        "authenticity_indicators": ["model numbers", "brand logos", "functionality"]
    },
    
    "oven": {
        "info": "Cooking appliance.",
        "value": "Professional and smart ovens command premium.",
        "tip": "Check heating elements and temperature accuracy.",
        "brands": ["Wolf", "Viking", "GE", "Whirlpool", "Bosch"],
        "conditions": ["new", "used", "commercial", "smart", "needs_repair"],
        "materials": ["stainless_steel", "glass", "enamel"],
        "estimated_value": "$200 - $10,000",
        "maintenance": "Clean regularly, check seals, calibrate temperature",
        "rarity": "Common",
        "authenticity_indicators": ["brand quality", "features", "construction"]
    },
    
    "toaster": {
        "info": "Appliance for toasting bread.",
        "value": "Vintage and designer toasters are collectible.",
        "tip": "Check heating elements and timer functionality.",
        "brands": ["Dualit", "Smeg", "Breville", "Cuisinart", "Sunbeam"],
        "conditions": ["new", "used", "vintage", "designer", "broken"],
        "materials": ["chrome", "stainless_steel", "plastic"],
        "estimated_value": "$20 - $500",
        "maintenance": "Clean crumb tray, descale if needed",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "design features", "functionality"]
    },
    
    "sink": {
        "info": "Plumbing fixture for washing.",
        "value": "Designer and antique sinks are valuable.",
        "tip": "Check for cracks and finish condition.",
        "brands": ["Kohler", "American Standard", "Elkay", "Franke", "Antique"],
        "conditions": ["new", "used", "antique", "farmhouse", "needs_repair"],
        "materials": ["porcelain", "stainless_steel", "copper", "granite"],
        "estimated_value": "$100 - $5,000",
        "maintenance": "Clean regularly, avoid abrasive cleaners, check seals",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "material quality", "construction"]
    },
    
    "refrigerator": {
        "info": "Appliance for cooling food.",
        "value": "Smart and commercial refrigerators command premium.",
        "tip": "Check cooling efficiency and door seals.",
        "brands": ["Sub-Zero", "Viking", "GE", "LG", "Samsung"],
        "conditions": ["new", "used", "commercial", "smart", "needs_repair"],
        "materials": ["stainless_steel", "plastic", "glass"],
        "estimated_value": "$300 - $20,000",
        "maintenance": "Clean coils, check seals, defrost if needed",
        "rarity": "Common",
        "authenticity_indicators": ["brand quality", "features", "energy efficiency"]
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
    },
    
    "scissors": {
        "info": "Cutting tool with two blades.",
        "value": "Antique and specialized scissors are collectible.",
        "tip": "Check for sharpness and pivot tightness.",
        "brands": ["Fiskars", "Gingher", "Kai", "Victorinox", "Antique"],
        "conditions": ["new", "used", "antique", "specialized", "dull"],
        "materials": ["steel", "stainless_steel", "titanium"],
        "estimated_value": "$5 - $500",
        "maintenance": "Clean after use, sharpen regularly, oil pivot",
        "rarity": "Common",
        "authenticity_indicators": ["brand markings", "steel quality", "construction"]
    },
    
    "teddy bear": {
        "info": "Stuffed toy bear.",
        "value": "Antique and limited edition bears are valuable.",
        "tip": "Check for original tags and condition.",
        "brands": ["Steiff", "Gund", "Build-A-Bear", "Vintage"],
        "conditions": ["new", "used", "antique", "collectible", "loved"],
        "materials": ["mohair", "plush", "felt", "cotton"],
        "estimated_value": "$10 - $10,000+",
        "maintenance": "Surface clean, avoid washing, store properly",
        "rarity": "Common to Rare",
        "authenticity_indicators": ["ear buttons", "tags", "materials"]
    },
    
    "hair drier": {
        "info": "Appliance for drying hair.",
        "value": "Professional and vintage models have value.",
        "tip": "Check heating elements and motor.",
        "brands": ["Dyson", "Bio Ionic", "Babyliss", "Revlon", "Vintage"],
        "conditions": ["new", "used", "professional", "vintage", "broken"],
        "materials": ["plastic", "metal", "ceramic"],
        "estimated_value": "$20 - $500",
        "maintenance": "Clean filter, check cord, store properly",
        "rarity": "Common",
        "authenticity_indicators": ["brand quality", "features", "wattage"]
    },
    
    "toothbrush": {
        "info": "Oral hygiene tool.",
        "value": "Electric and specialized brushes have value.",
        "tip": "Check bristle condition and battery.",
        "brands": ["Oral-B", "Philips Sonicare", "Quip", "Curaprox"],
        "conditions": ["new", "used", "electric", "manual", "worn"],
        "materials": ["plastic", "nylon", "bamboo"],
        "estimated_value": "$2 - $200",
        "maintenance": "Replace heads regularly, clean handle",
        "rarity": "Common",
        "authenticity_indicators": ["brand quality", "features", "authenticity"]
    },
    
    "hair brush": {
        "info": "Tool for hair care.",
        "value": "Natural bristle and antique brushes are valuable.",
        "tip": "Check bristle condition and handle.",
        "brands": ["Mason Pearson", "Wet Brush", "Tangle Teezer", "Antique"],
        "conditions": ["new", "used", "natural_bristle", "antique", "worn"],
        "materials": ["wood", "plastic", "boar_bristle", "nylon"],
        "estimated_value": "$5 - $300",
        "maintenance": "Clean regularly, dry properly, store carefully",
        "rarity": "Common",
        "authenticity_indicators": ["bristle type", "handle material", "brand"]
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
    Enhanced visual feature analysis with comprehensive assessment
    """
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure bbox is within frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    
    # Check if bbox is valid
    if x2 <= x1 or y2 <= y1:
        return None
    
    obj_image = frame[y1:y2, x1:x2]
    
    if obj_image.size == 0:
        return None
    
    features = {}
    
    # 1. Color Analysis (Dominant Colors)
    pixels = obj_image.reshape(-1, 3)
    
    # Find dominant colors using clustering
    if len(pixels) > 10:
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            features['dominant_colors'] = dominant_colors.tolist()
            
            # Color names
            color_names = []
            for color in dominant_colors:
                color_names.append(rgb_to_color_name(color))
            features['color_names'] = color_names
            
            # Color statistics
            features['color_mean'] = np.mean(pixels, axis=0).tolist()
            features['color_std'] = np.std(pixels, axis=0).tolist()
        except:
            features['dominant_colors'] = []
            features['color_names'] = []
    
    # 2. Texture Analysis
    gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
    
    # GLCM texture features
    glcm_features = calculate_glcm_features(gray)
    features.update(glcm_features)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    features['edge_density'] = float(edge_density)
    
    # 3. Shape Analysis
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        shape_features = analyze_contour(largest_contour)
        features.update(shape_features)
    
    # 4. Material Indicators
    features['material_indicators'] = analyze_material_indicators(obj_image)
    
    # 5. Condition Indicators
    features['condition_indicators'] = analyze_condition_indicators(obj_image, object_type)
    
    # 6. Size and Position
    features['relative_size'] = (obj_image.shape[0] * obj_image.shape[1]) / (frame.shape[0] * frame.shape[1]) if frame.shape[0] * frame.shape[1] > 0 else 0
    features['aspect_ratio'] = obj_image.shape[1] / obj_image.shape[0] if obj_image.shape[0] > 0 else 0
    
    # 7. Brightness and Contrast
    features['brightness'] = float(np.mean(gray))
    features['contrast'] = float(np.std(gray))
    
    # 8. Sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features['sharpness'] = float(laplacian_var)
    
    return features

def rgb_to_color_name(rgb):
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

def calculate_glcm_features(gray):
    """Calculate Gray Level Co-occurrence Matrix features"""
    import numpy as np
    
    # Ensure 8-bit
    if gray.dtype != np.uint8:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    
    # Ensure not empty
    if gray.size == 0:
        return {
            'contrast': 0.0,
            'homogeneity': 0.0,
            'energy': 0.0,
            'correlation': 0.0
        }
    
    # Calculate GLCM
    try:
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extract features
        features = {
            'contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
            'homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
            'energy': float(graycoprops(glcm, 'energy')[0, 0]),
            'correlation': float(graycoprops(glcm, 'correlation')[0, 0])
        }
    except:
        features = {
            'contrast': 0.0,
            'homogeneity': 0.0,
            'energy': 0.0,
            'correlation': 0.0
        }
    
    return features

def analyze_contour(contour):
    """Analyze shape features from contour"""
    import cv2
    
    features = {}
    
    # Basic shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    features['area'] = float(area)
    features['perimeter'] = float(perimeter)
    
    # Shape descriptors
    if perimeter > 0:
        features['circularity'] = float(4 * np.pi * area / (perimeter * perimeter))
    
    # Bounding box ratio
    x, y, w, h = cv2.boundingRect(contour)
    features['rectangularity'] = float(area / (w * h)) if w * h > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features['solidity'] = float(area / hull_area) if hull_area > 0 else 0
    
    # Aspect ratio of bounding box
    features['bbox_aspect_ratio'] = float(w / h) if h > 0 else 0
    
    return features

def analyze_material_indicators(image):
    """Analyze visual indicators of material"""
    import cv2
    import numpy as np
    
    indicators = {}
    
    # Reflectivity (brightness variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_variance = np.var(gray)
    
    if brightness_variance > 5000:
        indicators['reflectivity'] = 'high'
        indicators['possible_materials'] = ['metal', 'glass', 'ceramic']
    elif brightness_variance > 2000:
        indicators['reflectivity'] = 'medium'
        indicators['possible_materials'] = ['plastic', 'painted_surface', 'glazed_ceramic']
    else:
        indicators['reflectivity'] = 'low'
        indicators['possible_materials'] = ['wood', 'fabric', 'paper', 'matte_surface']
    
    # Color consistency (for uniform materials)
    color_std = np.std(image.reshape(-1, 3), axis=0)
    mean_std = np.mean(color_std)
    
    if mean_std < 20:
        indicators['color_consistency'] = 'high'
    elif mean_std < 40:
        indicators['color_consistency'] = 'medium'
    else:
        indicators['color_consistency'] = 'low'
    
    # Edge patterns (for textured materials)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    if edge_density > 0.1:
        indicators['texture_level'] = 'high'
        indicators['texture_type'] = 'detailed_pattern'
    elif edge_density > 0.05:
        indicators['texture_level'] = 'medium'
        indicators['texture_type'] = 'moderate_pattern'
    else:
        indicators['texture_level'] = 'low'
        indicators['texture_type'] = 'smooth'
    
    # Surface smoothness
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detail = cv2.absdiff(gray, blur)
    detail_level = np.mean(detail)
    
    if detail_level > 30:
        indicators['surface_smoothness'] = 'rough'
    elif detail_level > 15:
        indicators['surface_smoothness'] = 'medium'
    else:
        indicators['surface_smoothness'] = 'smooth'
    
    return indicators

def analyze_condition_indicators(image, object_type):
    """Analyze condition based on visual cues"""
    import cv2
    import numpy as np
    
    indicators = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Scratches/damage detection
    edges = cv2.Canny(gray, 30, 100)
    scratch_ratio = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    # Color fading (variance in saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_variance = np.var(hsv[:, :, 1])
    
    # Dust/dirt detection (low frequency patterns)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detail = cv2.absdiff(gray, blur)
    detail_level = np.mean(detail)
    
    # Stain detection (color anomalies)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_std, b_std = np.std(a), np.std(b)
    color_anomaly = (a_std + b_std) / 2
    
    # Assign condition scores
    condition_score = 10  # Start with perfect
    
    # Deduct for scratches
    if scratch_ratio > 0.15:
        condition_score -= 4
        indicators['scratches'] = 'heavy'
        indicators['scratch_severity'] = 'severe'
    elif scratch_ratio > 0.08:
        condition_score -= 2
        indicators['scratches'] = 'moderate'
        indicators['scratch_severity'] = 'moderate'
    elif scratch_ratio > 0.03:
        condition_score -= 1
        indicators['scratches'] = 'light'
        indicators['scratch_severity'] = 'minor'
    else:
        indicators['scratches'] = 'none'
        indicators['scratch_severity'] = 'none'
    
    # Deduct for fading
    if saturation_variance < 100:
        condition_score -= 2
        indicators['fading'] = 'yes'
        indicators['color_vibrancy'] = 'faded'
    elif saturation_variance < 200:
        condition_score -= 1
        indicators['fading'] = 'slight'
        indicators['color_vibrancy'] = 'moderate'
    else:
        indicators['fading'] = 'no'
        indicators['color_vibrancy'] = 'vibrant'
    
    # Deduct for dust/dirt
    if detail_level > 30:
        condition_score -= 3
        indicators['cleanliness'] = 'dirty'
        indicators['surface_cleanliness'] = 'poor'
    elif detail_level > 15:
        condition_score -= 1
        indicators['cleanliness'] = 'dusty'
        indicators['surface_cleanliness'] = 'fair'
    else:
        indicators['cleanliness'] = 'clean'
        indicators['surface_cleanliness'] = 'good'
    
    # Deduct for stains
    if color_anomaly > 20:
        condition_score -= 2
        indicators['stains'] = 'present'
        indicators['stain_severity'] = 'noticeable'
    elif color_anomaly > 10:
        condition_score -= 1
        indicators['stains'] = 'slight'
        indicators['stain_severity'] = 'minor'
    else:
        indicators['stains'] = 'none'
        indicators['stain_severity'] = 'none'
    
    # Object-type specific deductions
    if object_type in ['car', 'motorcycle', 'bicycle']:
        # Check for dents (localized dark areas)
        local_darkness = cv2.filter2D(gray, -1, np.ones((5,5))/25)
        local_variance = np.var(local_darkness)
        if local_variance > 500:
            condition_score -= 1
            indicators['dents'] = 'possible'
    
    elif object_type in ['electronic', 'laptop', 'cell phone', 'tv']:
        # Check for screen damage
        screen_region = image[int(image.shape[0]*0.1):int(image.shape[0]*0.9),
                            int(image.shape[1]*0.1):int(image.shape[1]*0.9)]
        if screen_region.size > 0:
            screen_brightness = np.mean(cv2.cvtColor(screen_region, cv2.COLOR_BGR2GRAY))
            if screen_brightness < 50:
                condition_score -= 2
                indicators['screen_condition'] = 'dark_spots'
    
    elif object_type in ['furniture', 'chair', 'couch', 'table']:
        # Check for upholstery wear
        texture_variance = np.var(gray)
        if texture_variance > 1000:
            condition_score -= 1
            indicators['upholstery_wear'] = 'visible'
    
    # Ensure score stays within bounds
    condition_score = max(0, min(10, condition_score))
    indicators['condition_score'] = condition_score
    
    # Overall condition assessment
    if condition_score >= 9:
        indicators['overall_condition'] = 'excellent'
        indicators['condition_description'] = 'Like new, minimal wear'
    elif condition_score >= 7:
        indicators['overall_condition'] = 'good'
        indicators['condition_description'] = 'Normal wear, fully functional'
    elif condition_score >= 5:
        indicators['overall_condition'] = 'fair'
        indicators['condition_description'] = 'Visible wear, needs attention'
    else:
        indicators['overall_condition'] = 'poor'
        indicators['condition_description'] = 'Significant wear/damage, needs repair'
    
    # Maintenance recommendations
    if condition_score < 7:
        if 'scratches' in indicators and indicators['scratches'] != 'none':
            indicators['recommended_maintenance'] = 'Polish/refinish surface'
        elif 'cleanliness' in indicators and indicators['cleanliness'] in ['dusty', 'dirty']:
            indicators['recommended_maintenance'] = 'Deep cleaning recommended'
        elif 'fading' in indicators and indicators['fading'] != 'no':
            indicators['recommended_maintenance'] = 'Consider restoration/repainting'
        else:
            indicators['recommended_maintenance'] = 'Regular maintenance needed'
    else:
        indicators['recommended_maintenance'] = 'Continue regular care'
    
    return indicators

def estimate_condition_from_features(features, object_type):
    """
    Estimate object condition based on visual features
    """
    if not features:
        return "unknown", 0.5
    
    # Get condition from analysis if available
    if 'condition_indicators' in features:
        condition_info = features['condition_indicators']
        condition = condition_info.get('overall_condition', 'unknown')
        score = condition_info.get('condition_score', 5) / 10.0
        return condition, score
    
    # Fallback to feature-based estimation
    condition_rules = {
        "car": {
            "excellent": features.get('sharpness', 0) > 100 and features.get('color_variance', 0) < 30,
            "good": features.get('sharpness', 0) > 50,
            "fair": features.get('sharpness', 0) > 20,
            "poor": features.get('sharpness', 0) <= 20
        },
        "electronic": {
            "excellent": features.get('sharpness', 0) > 150,
            "good": features.get('sharpness', 0) > 80,
            "fair": features.get('sharpness', 0) > 40,
            "poor": features.get('sharpness', 0) <= 40
        },
        "furniture": {
            "excellent": features.get('texture_density', 0) < 0.05,
            "good": features.get('texture_density', 0) < 0.1,
            "fair": features.get('texture_density', 0) < 0.2,
            "poor": features.get('texture_density', 0) >= 0.2
        },
        "default": {
            "excellent": features.get('sharpness', 0) > 100 and features.get('edge_density', 0) < 0.05,
            "good": features.get('sharpness', 0) > 50,
            "fair": features.get('sharpness', 0) > 20,
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

def comprehensive_object_assessment(object_type, bbox, frame, confidence):
    """
    Perform comprehensive assessment of an object
    """
    # Visual analysis
    features = analyze_object_visual_features(frame, bbox, object_type)
    
    # Get knowledge base info
    if object_type in ENHANCED_KNOWLEDGE:
        knowledge = ENHANCED_KNOWLEDGE[object_type]
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
    
    # Estimate value
    value_estimate = estimate_object_value(object_type, condition, features)
    
    # Generate assessment
    assessment = {
        "identification": {
            "object_type": object_type,
            "confidence": confidence,
            "alternatives": suggest_similar_objects(object_type, features)
        },
        "condition": {
            "rating": condition,
            "score": condition_info.get('condition_score', 5),
            "details": condition_info
        },
        "visual_characteristics": {
            "colors": features.get('color_names', ['unknown']) if features else ['unknown'],
            "texture": features.get('material_indicators', {}).get('texture_level', 'unknown') if features else 'unknown',
            "material_indicators": features.get('material_indicators', {}) if features else {},
            "size": {
                "relative": features.get('relative_size', 0) if features else 0,
                "aspect_ratio": features.get('aspect_ratio', 0) if features else 0
            }
        },
        "value_assessment": {
            "estimated_value": value_estimate,
            "factors": get_value_factors(object_type, condition, features),
            "potential_appreciation": estimate_appreciation_potential(object_type, condition)
        },
        "knowledge_base": {
            "info": knowledge["info"],
            "common_brands": knowledge.get("brands", ["Various"]),
            "typical_materials": knowledge.get("materials", ["Unknown"]),
            "expert_tip": knowledge.get("tip", ""),
            "maintenance_guidance": knowledge.get("maintenance", "Handle with care")
        },
        "recommendations": {
            "maintenance": knowledge.get('maintenance', 'Handle with care'),
            "storage": get_storage_recommendations(object_type, condition),
            "cleaning": get_cleaning_recommendations(object_type),
            "care_immediate": condition_info.get('recommended_maintenance', 'Regular care') if condition_info else 'Regular care'
        },
        "authentication": {
            "tips": knowledge.get('authenticity_indicators', ['Check overall quality']),
            "common_fakes": get_common_fakes(object_type),
            "verification_methods": get_verification_methods(object_type)
        },
        "market_info": {
            "rarity": knowledge.get('rarity', 'unknown'),
            "demand_level": get_demand_level(object_type),
            "best_selling_platforms": get_selling_platforms(object_type),
            "value_trend": get_value_trend(object_type)
        }
    }
    
    return assessment

def estimate_object_value(object_type, condition, features):
    """Estimate object value based on multiple factors"""
    # Base values from knowledge base
    base_values = {
        "car": 20000,
        "laptop": 800,
        "cell phone": 500,
        "bicycle": 300,
        "motorcycle": 5000,
        "handbag": 200,
        "backpack": 50,
        "chair": 150,
        "couch": 800,
        "table": 300,
        "bottle": 25,
        "book": 20,
        "clock": 100,
        "vase": 75,
        "default": 100
    }
    
    base_value = base_values.get(object_type, base_values["default"])
    
    # Apply condition multiplier
    condition_multipliers = {
        'excellent': 1.0,
        'good': 0.7,
        'fair': 0.4,
        'poor': 0.2,
        'unknown': 0.5
    }
    
    multiplier = condition_multipliers.get(condition, 0.5)
    adjusted_value = base_value * multiplier
    
    # Apply material multiplier if known
    if features and 'material_indicators' in features:
        materials = features['material_indicators'].get('possible_materials', [])
        if materials:
            # Take the highest value material
            max_material_multiplier = 1.0
            for material in materials:
                if material in MATERIAL_VALUE_IMPACT:
                    max_material_multiplier = max(max_material_multiplier, MATERIAL_VALUE_IMPACT[material])
            adjusted_value *= max_material_multiplier
    
    # Apply rarity factor
    if object_type in ENHANCED_KNOWLEDGE:
        rarity = ENHANCED_KNOWLEDGE[object_type].get('rarity', 'common')
        rarity_multipliers = {
            'ultra_rare': 10.0,
            'rare': 5.0,
            'uncommon': 2.0,
            'common': 1.0,
            'protected': 0.0  # Not for sale
        }
        adjusted_value *= rarity_multipliers.get(rarity, 1.0)
    
    # Format value
    if adjusted_value >= 1000000:
        return f"${adjusted_value/1000000:.2f}M"
    elif adjusted_value >= 1000:
        return f"${adjusted_value/1000:.1f}K"
    else:
        return f"${adjusted_value:,.2f}"

def suggest_similar_objects(object_type, features):
    """Suggest similar objects for verification"""
    similarity_map = {
        "laptop": ["notebook", "tablet", "computer"],
        "cell phone": ["smartphone", "mobile phone", "iphone"],
        "car": ["vehicle", "automobile", "truck", "van"],
        "bottle": ["glass", "container", "jar", "flask"],
        "chair": ["stool", "seat", "bench", "ottoman"],
        "book": ["notebook", "magazine", "journal"],
        "clock": ["watch", "timer", "chronometer"],
        "vase": ["urn", "pot", "jar", "vessel"]
    }
    return similarity_map.get(object_type, [])

def get_value_factors(object_type, condition, features):
    """Get factors affecting value"""
    factors = []
    
    factors.append(f"Condition: {condition}")
    
    if features and 'material_indicators' in features:
        material_info = features['material_indicators']
        if 'possible_materials' in material_info:
            factors.append(f"Possible materials: {', '.join(material_info['possible_materials'][:3])}")
        
        if 'reflectivity' in material_info:
            factors.append(f"Surface reflectivity: {material_info['reflectivity']}")
    
    if object_type in ENHANCED_KNOWLEDGE:
        knowledge = ENHANCED_KNOWLEDGE[object_type]
        factors.append(f"Rarity: {knowledge.get('rarity', 'unknown')}")
        
        brands = knowledge.get('brands', ['Various'])
        if brands != ['Various']:
            factors.append(f"Brand premium: Yes (common brands: {', '.join(brands[:3])})")
    
    # Size factor
    if features and 'relative_size' in features:
        size = features['relative_size']
        if size > 0.3:
            factors.append("Size: Large (premium for some items)")
        elif size < 0.05:
            factors.append("Size: Small (portable/collectible)")
    
    return factors

def estimate_appreciation_potential(object_type, condition):
    """Estimate if object value might appreciate"""
    appreciation_keywords = ["antique", "vintage", "collectible", "art", "luxury", "limited", "rare"]
    depreciation_keywords = ["electronic", "phone", "laptop", "modern"]
    
    object_lower = object_type.lower()
    
    for keyword in appreciation_keywords:
        if keyword in object_lower:
            return "High appreciation potential if well-maintained"
    
    for keyword in depreciation_keywords:
        if keyword in object_lower:
            return "Likely depreciating asset (technology items)"
    
    if condition in ['excellent', 'good'] and object_type in ['car', 'motorcycle', 'furniture']:
        return "May hold value with proper care"
    
    if condition == 'excellent' and object_type in ['handbag', 'watch', 'jewelry']:
        return "Good chance of maintaining value"
    
    return "Variable - depends on market and maintenance"

def get_storage_recommendations(object_type, condition):
    """Get storage recommendations"""
    recommendations = {
        "excellent": "Store in controlled environment away from direct sunlight",
        "good": "Keep in clean, dry place with stable temperature",
        "fair": "Handle with care, consider protective casing",
        "poor": "Consider professional restoration before long-term storage"
    }
    
    base = recommendations.get(condition, "Handle with care")
    
    # Object-specific additions
    if "electronic" in object_type.lower() or object_type in ["laptop", "cell phone", "tv"]:
        return f"{base}. Keep battery at 50% charge for long-term storage. Remove batteries if possible."
    elif "paper" in object_type.lower() or object_type in ["book", "magazine"]:
        return f"{base}. Store upright, use acid-free materials if valuable. Avoid humidity."
    elif "fabric" in object_type.lower() or object_type in ["clothing", "furniture"]:
        return f"{base}. Use breathable covers, avoid plastic. Protect from moths."
    elif "wood" in object_type.lower() or object_type in ["furniture", "chair", "table"]:
        return f"{base}. Maintain stable humidity (40-60%). Use furniture wax/polish."
    elif "metal" in object_type.lower():
        return f"{base}. Control humidity to prevent rust. Use protective coatings if needed."
    
    return base

def get_cleaning_recommendations(object_type):
    """Get cleaning recommendations"""
    if "electronic" in object_type.lower() or object_type in ["laptop", "cell phone"]:
        return "Use microfiber cloth and isopropyl alcohol (70%) for screens and surfaces. Avoid liquids near ports."
    elif object_type in ["car", "motorcycle", "bicycle"]:
        return "Professional detailing recommended for best results. Regular washing with appropriate cleaners."
    elif object_type in ["handbag", "backpack", "clothing"]:
        return "Use appropriate leather/fabric cleaner based on material. Test on small area first."
    elif "wood" in object_type.lower():
        return "Dust regularly with soft cloth. Use wood-specific cleaners. Avoid harsh chemicals."
    elif "glass" in object_type.lower() or object_type in ["vase", "bottle"]:
        return "Clean with mild soap and water. Dry thoroughly to prevent water spots."
    else:
        return "Gentle cleaning with appropriate materials. Avoid abrasive cleaners."

def get_common_fakes(object_type):
    """Get common fake indicators for this object type"""
    fakes = {
        "handbag": ["logo misalignment", "poor stitching quality", "incorrect serial format", "wrong materials"],
        "watch": ["incorrect weight", "poor movement quality", "wrong materials", "misspelled brand"],
        "painting": ["wrong brush stroke patterns", "incorrect aging signs", "poor canvas quality", "wrong pigments"],
        "jewelry": ["incorrect markings", "poor stone setting", "wrong metal composition", "missing hallmarks"],
        "electronic": ["non-original parts", "wrong serial numbers", "poor build quality", "incorrect packaging"],
        "default": ["poor craftsmanship", "materials don't match description", "wrong markings", "suspiciously low price"]
    }
    
    return fakes.get(object_type, fakes["default"])

def get_verification_methods(object_type):
    """Get verification methods"""
    methods = {
        "handbag": ["serial number check with brand", "authenticity card verification", "material analysis", "hardware examination"],
        "watch": ["movement inspection by expert", "serial number verification", "weight check", "case back examination"],
        "painting": ["UV light inspection", "canvas thread analysis", "pigment analysis", "expert appraisal"],
        "jewelry": ["gemstone certification", "metal purity test", "hallmark verification", "professional appraisal"],
        "electronic": ["serial number check with manufacturer", "software authenticity check", "part verification", "performance testing"],
        "default": ["visual inspection by expert", "weight and dimension check", "material testing", "documentation review"]
    }
    
    return methods.get(object_type, methods["default"])

def get_demand_level(object_type):
    """Get market demand level"""
    high_demand = ["cell phone", "laptop", "car", "handbag", "watch", "game console"]
    medium_demand = ["bicycle", "backpack", "chair", "clock", "furniture", "kitchen_appliances"]
    low_demand = ["vintage_items", "collectibles", "specialty_items", "antiques"]
    
    if object_type in high_demand:
        return "High"
    elif object_type in medium_demand:
        return "Medium"
    elif object_type in low_demand:
        return "Low (niche market)"
    else:
        return "Variable"

def get_selling_platforms(object_type):
    """Get best selling platforms"""
    platforms = {
        "electronics": ["eBay", "Swappa", "Gazelle", "Facebook Marketplace"],
        "luxury": ["The RealReal", "Vestiaire Collective", "1stDibs", "Fashionphile"],
        "collectibles": ["eBay", "Heritage Auctions", "Bonhams", "LiveAuctioneers"],
        "vehicles": ["AutoTrader", "Cars.com", "Facebook Marketplace", "Craigslist"],
        "furniture": ["Facebook Marketplace", "Craigslist", "OfferUp", "Chairish"],
        "general": ["Facebook Marketplace", "Craigslist", "OfferUp", "Nextdoor"],
        "default": ["eBay", "Facebook Marketplace", "local classifieds"]
    }
    
    if "phone" in object_type.lower() or "laptop" in object_type.lower():
        return platforms["electronics"]
    elif "handbag" in object_type.lower() or "watch" in object_type.lower() or "jewelry" in object_type.lower():
        return platforms["luxury"]
    elif "antique" in object_type.lower() or "vintage" in object_type.lower() or "collectible" in object_type.lower():
        return platforms["collectibles"]
    elif "car" in object_type.lower() or "motorcycle" in object_type.lower() or "bicycle" in object_type.lower():
        return platforms["vehicles"]
    elif "chair" in object_type.lower() or "couch" in object_type.lower() or "table" in object_type.lower():
        return platforms["furniture"]
    else:
        return platforms["default"]

def get_value_trend(object_type):
    """Get value trend information"""
    appreciating_items = ["antique", "vintage", "collectible", "art", "limited_edition", "luxury"]
    depreciating_items = ["electronic", "phone", "laptop", "modern_appliance", "fashion_trend"]
    
    object_lower = object_type.lower()
    
    for keyword in appreciating_items:
        if keyword in object_lower:
            return "Generally appreciating over time"
    
    for keyword in depreciating_items:
        if keyword in object_lower:
            return "Generally depreciating (technology/fashion)"
    
    return "Stable or variable based on condition and market"

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
    
    if 'brands' in report and report['brands'] != ["Various"]:
        lines.append(f"🏷️ Common Brands: {', '.join(report['brands'][:3])}")
    
    lines.append(f"🔍 Info: {report['info']}")
    
    if 'expert_tip' in report and report['expert_tip']:
        lines.append(f"💡 Tip: {report['expert_tip']}")
    
    if 'maintenance' in report:
        lines.append(f"🛠️ Maintenance: {report['maintenance']}")
    
    return lines

def format_comprehensive_assessment(assessment):
    """
    Format comprehensive assessment for display
    """
    lines = []
    
    # Header
    lines.append("="*50)
    lines.append(f"📊 COMPREHENSIVE ASSESSMENT: {assessment['identification']['object_type'].upper()}")
    lines.append("="*50)
    
    # Identification
    lines.append(f"🎯 Confidence: {assessment['identification']['confidence']:.2f}")
    
    # Condition
    lines.append(f"\n📊 CONDITION ASSESSMENT:")
    lines.append(f"   Rating: {assessment['condition']['rating'].upper()}")
    lines.append(f"   Score: {assessment['condition']['score']}/10")
    
    # Value
    lines.append(f"\n💰 VALUE ASSESSMENT:")
    lines.append(f"   Estimated Value: {assessment['value_assessment']['estimated_value']}")
    lines.append(f"   Appreciation Potential: {assessment['value_assessment']['potential_appreciation']}")
    
    # Visual Characteristics
    if assessment['visual_characteristics']['colors'] != ['unknown']:
        lines.append(f"\n🎨 VISUAL CHARACTERISTICS:")
        lines.append(f"   Colors: {', '.join(assessment['visual_characteristics']['colors'])}")
        lines.append(f"   Texture: {assessment['visual_characteristics']['texture']}")
    
    # Knowledge Base
    lines.append(f"\n📚 KNOWLEDGE BASE:")
    lines.append(f"   Info: {assessment['knowledge_base']['info']}")
    
    if assessment['knowledge_base']['common_brands'] != ["Various"]:
        lines.append(f"   Common Brands: {', '.join(assessment['knowledge_base']['common_brands'][:3])}")
    
    # Recommendations
    lines.append(f"\n💡 RECOMMENDATIONS:")
    lines.append(f"   Maintenance: {assessment['recommendations']['maintenance']}")
    lines.append(f"   Storage: {assessment['recommendations']['storage']}")
    lines.append(f"   Cleaning: {assessment['recommendations']['cleaning']}")
    
    if 'care_immediate' in assessment['recommendations']:
        lines.append(f"   Immediate Care: {assessment['recommendations']['care_immediate']}")
    
    # Market Info
    lines.append(f"\n📈 MARKET INFORMATION:")
    lines.append(f"   Rarity: {assessment['market_info']['rarity']}")
    lines.append(f"   Demand: {assessment['market_info']['demand_level']}")
    lines.append(f"   Value Trend: {assessment['market_info']['value_trend']}")
    
    # Best Platforms
    platforms = assessment['market_info']['best_selling_platforms']
    lines.append(f"   Best Selling Platforms: {', '.join(platforms[:3])}")
    
    lines.append("="*50)
    
    return lines
# frontend/llm_client.py
import os, re, json
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
from openai import OpenAI

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://ollama:11434/v1")
OLLAMA_KEY  = os.getenv("OLLAMA_KEY", "ollama")
MODEL       = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

client = OpenAI(base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)

# --- cleaners ---------------------------------------------------------------
THINK_BLOCK = re.compile(r"(?is)\s*<think>.*?</think>\s*")

def _strip_think(text: str) -> str:
    t = text or ""
    t = THINK_BLOCK.sub("", t)
    t = re.sub(r"```(?:json|markdown)?", "", t)
    t = re.sub(r"```", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# --- robust JSON extractor ---------------------------------------------------
def _balanced_json_slice(s: str):
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _extract_json(text: str) -> dict:
    if not text:
        return {}
    t = _strip_think(text)
    blob = _balanced_json_slice(t) or t
    blob = blob.replace("’","'").replace("“","\"").replace("”","\"")
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    try:
        return json.loads(blob)
    except Exception:
        try:
            blob2 = re.sub(r"(?<!\\)'", '"', blob)
            blob2 = re.sub(r",\s*([}\]])", r"\1", blob2)
            return json.loads(blob2)
        except Exception:
            return {}

# --- helpers ----------------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9&/\- ]", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _pick_best(query: str, options: List[str], threshold: float = 0.65) -> str:
    if not options:
        return ""
    q = _norm(query)
    best, best_r = "", 0.0
    for opt in options:
        r = _ratio(q, opt)
        if r > best_r:
            best, best_r = opt, r
    return best if best_r >= threshold else ""

def _pick_any_from_text(user_text: str, options: List[str]) -> str:
    txt = " " + _norm(user_text) + " "
    for opt in options:
        o = " " + _norm(opt) + " "
        if o in txt:
            return opt
    return ""

def _canon_object(obj: str) -> str:
    obj = (obj or "").strip().lower()
    if obj.endswith("es") and not obj.endswith("ses"):
        obj = obj[:-2]
    elif obj.endswith("s") and not obj.endswith("ss"):
        obj = obj[:-1]
    return obj

# --- STRICT CATEGORY CLASSIFIER (kept exactly as you provided) --------------
def extract_structured_with_taxonomy(
    user_text: str,
    cat1_options: List[str],
    cat2_options: List[str],
    cat3_options: List[str],
    brand_options: Optional[List[str]] = None,
) -> dict:
    """
    Uses your exact strict Category Classifier prompt with the user_text appended at the end.
    """
    brand_options = brand_options or []

    # EXACT prompt with user's input appended at the end
    system = f"""You are a strict product Category Classifier.

Allowed labels

Return exactly one label from this list (copy it exactly, including capitalization and punctuation):
['Auto', 'Home', 'Outdoor Living & Garden', 'Baby & Kids', 'Clothing, Shoes & Accessories', 'Handbags & Jewelry', 'Lifestyle & Recreation', 'Electronics', 'Toys & Baby', 'Fashion & Beauty', 'Kitchen & Dining', 'Beauty', 'Lifestyle', 'Fashion', 'Pet', 'Audio/Video', 'Lawn & Garden', 'Tires', 'Recreation', 'Shoes', 'Outdoor Furniture & Decor', 'Lighting', 'Fashion Jewelry', 'Wearable Technology', 'Rugs & Decor', "Women's Clothing", 'Toys', 'Furniture', 'Women', 'Computers', 'Glassware & Barware', 'Baby', 'Handbags & Wallets', 'Personal Audio & Video', 'Fragrance', 'Household Essentials', 'Dinnerware & Serveware', 'Hobbies & Crafts', 'Cookware & Bakeware', 'Hair Care', 'Watches', 'TVs & Entertainment', 'Personal Care', "Men's Clothing", 'Outdoor Tools & Equipment', 'Grills & Outdoor Cooking', 'Kitchen Appliances', 'Furniture & Decor', 'Bedding & Bath', 'Video Games', 'Fine Jewelry', 'Sunglasses', 'Heating & Cooling', 'Home Improvement', 'Large Appliances', 'Men', 'Sports', 'Flatware & Cutlery', 'Nail Care', 'Kitchen Tools & Utensils', 'Household Basics', 'Kids', 'Floorcare', 'Kitchen & Bath', 'Health & Fitness', 'Basics', 'Pools & Spas', 'Storage & Organization', 'Makeup', 'Skin Care', 'Cat', 'Outdoor', "Kids' Clothing", 'Housewares', 'Wedding & Engagement', 'Music', 'Kitchen Storage & Organization', 'Accessories', 'Dog', 'Kitchen Linens', 'Outdoors', 'Car Seats', 'Pet Essentials', 'Bed & Bath', 'Dash Cams', 'Pots & Planters', 'Light Truck/SUV', 'Bikes & Scooters', "Men's Shoes", 'Outdoor Furniture', 'Table & Floor Lamps', 'Earrings', 'Hunting & Fishing', 'Activity Trackers', 'Rugs', 'Activewear', 'Dress Up & Pretend Play', 'Hall & Entry Furniture', 'Clothing', 'Accent Pillows & Blankets', 'Passenger ', 'iPads & Tablets', 'Small Animal', 'Riding Toys', 'Drinkware', 'Strollers & Carriers', 'Luggage & Travel', 'Smart Home', 'Perfume', 'Smart Watches', 'Cleaning Essentials', 'Necklaces', 'Dinnerware', 'Pots & Pans', 'Travel', 'Hair Products', "Women's", 'Home Audio', 'Sensual Wellness', 'Handbags', 'Outerwear', 'Vehicles & Remote Control Toys', "Women's Shoes", 'Camping & Hiking', "Men's Grooming", 'Televisions', 'Coolers', 'Grills & Smokers', 'Passenger', 'Toasters', 'Tops', 'Holiday Decor', 'Bedroom', 'Living Room Furniture', 'Dolls & Dollhouses', 'Headphones', 'Cameras', 'Desktops', 'Portables', 'Learning Toys', 'Outdoor Play', 'Bedding', 'Grilling Tools & Accessories', 'Kitchen & Dining Room Furniture', 'Cologne', 'Fine Jewelry Sets', "Men's Sunglasses", 'Projectors & Screens', 'Food Processors', 'Laptops', 'Smartphones & iPods', 'Trailer', 'Rings', 'Games', 'Living Room', 'Home Office Furniture', 'Air Conditioners', 'Window Treatments', 'Tools', 'Notebooks', ' Monitors', "Kids' Crafts", 'Heaters', 'Baby Toys', 'Laundry', 'Air Purifiers & Dehumidifiers', 'Bracelets', 'Soccer', 'Knives & Cutting Boards', 'Pressure Cookers', 'Nail Polish', 'Mirrors', 'Coats, Jackets & Vests', 'Coffee & Tea', "Men's Jewelry", 'Decorative Objects', 'Bath Towels & Mats', 'Jewelry', 'Blocks & Building Toys', 'Specialty Tools', 'Bedroom Furniture', 'General Home', 'Gadgets & Specialty', "Women's Sunglasses", 'Hospital Uniforms', 'Vacuums', 'Pressure Washers', 'Food', 'Exercise', 'Cookware Sets', 'Crafts', 'Plush Toys & Puppets', 'Streaming Devices', "Boys' Shoes", 'Health & Safety', 'Wallets', "Men's", 'Sweaters & Hoodies', 'Decor', 'Pantry', 'Crafting', 'Bath & Body', 'Slow Cookers & Roasters', 'Baking Sheets & Dishes', 'Dining Room', 'Bakeware Sets', 'Socks & Underwear', 'Pool Toys & Floats', 'Bathroom Storage & Organization', 'Air Fryers & Deep Fryers', 'Mixers', 'Office & Accessories', 'Blenders', 'Serums & Treatments', 'Health', 'Oral Care', 'Golf', 'Hedgers & Trimmers', 'Self Care & Recovery', 'Water Sports', 'Moisturizers', 'Fine Necklaces', 'Home Office', 'Wireless Networking', 'Wall Decor', 'Bottoms', 'Pillows', 'Consoles', 'Socks & Intimates', 'Action Figures & Playsets', 'Charms', 'Fans', 'Chargers & Batteries', 'Gaming Notebooks', 'Ceiling Lights', 'Fire Pits & Heaters', 'Pet Supplies', 'Styling Tools', 'Flatware Sets', 'Baby & Kids Furniture', 'Lawn Mowers & Leaf Blowers', 'Blu-Ray Players', 'Closet Organization', 'Fine Earrings', 'Kitchen Utensils', 'Microwaves', 'Ranges', 'Colanders & Strainers', "Kids' Electronics", 'Towel Warmers & Bath Accessories', 'Learning', 'Jewelry Sets', 'PC Gaming', 'Chain Saws & Pole Saws', 'GPS Trackers', 'Dishwashers', 'Parts & Accessories', 'Irons & Steamers', 'Sun Care', 'Duffels & Overnight Bags', 'Mattresses', 'Baseball & Softball', 'Peelers & Choppers', 'Pizza Ovens', "Women's Grooming", 'Refrigeration', 'Outdoor Decor', 'Engagement Rings', 'Baking Accessories', 'Laundry Storage & Organization', "Kid's Shoes", 'Fine Bracelets', 'Household Products', 'Fine Rings', 'Pool Tools & Care', 'Cleansers', 'Pools', "Boys' Clothing", 'Keyboards & Synthesizers', 'Serveware', 'Food Storage Bags & Containers', 'Shower Curtains & Liners', 'Generators', 'Processing & Prep', 'Built-In Cooking', 'Baby Shoes', 'Bags & Wallets', 'Drones', 'Laundry & Cabinet Organization', 'Juicers', 'Sink & Countertop Organizers', 'Eyes', 'Basketball', 'Kitchen', 'Wigs', 'Virtual Reality', 'Car Audio', 'Sleepwear & Loungewear', 'Skincare', 'Aromatherapy & Relaxation', 'Pretend Play', 'Playroom', 'Tennis', 'Skin Care Tools', 'Freezers', 'Football', 'Dog Supplies & Care', 'Greenhouses & Gardening Tools', 'Mattresses & Pads', 'Baby Accessories', 'Feeding', 'Playmats & Activity Centers', 'Face', 'Karaoke Machines', 'Beds & Furniture', 'Bar Tools & Accessories', 'GPS / Navigation', 'Storage Bins & Baskets', 'Snow Removal', 'Microwaves & Toasters', 'Paper & Plastic', 'Home Improvement Accessories', 'Wedding Bands', 'Animals & Puppets', 'Lawn Care', 'Home Theater Systems', 'Printers', 'Coffee Storage & Accessories', 'Pickleball', 'Laundry Essentials', 'Tillers & Cultivators', 'Sheds & Storage', 'Dog Toys', 'Kitchen Towels & Napkins', 'Placemats & Trivets', 'Outdoor Lighting', 'Luggage', 'Bathroom', 'Graters & Zesters', 'Camping', 'Action Cameras', ' Haircare', 'Range Hoods', 'Lightbulbs & Electrical', 'Carpet Cleaners', 'Outdoor Cooking', 'Bath', 'Spas', 'Sinks', 'Firepits', 'Receiver', 'Racing', 'Tools & Brushes', 'Measuring Cups & Spoons', 'Mats/Cleaning', 'Lips', 'Steam Cleaners', 'Gates & Fencing Systems', 'iPods & MP3 Players', 'Dog Beds & Crates', 'Kitchen Cabinet & Drawer Organizers', 'Food Scales', 'Speakers', 'Baby Toiletries', 'Health & First Aid', 'Apparel & Accessories', "Girls' Clothing", 'Sweepers & Dusters', 'Motorcycle', 'Navigation', 'Trucks & Trains', 'Kitchen Timers', 'Nursery', 'Office Organization', 'Telecommunication', 'Food & Treats', 'Hobbies', 'Drawer & Cabinet Organization', 'Pools & Floats', 'Supplies & Care', 'Lighting & Decor', 'Remote Control Toys', 'Bikes & Cycling', 'Cat Beds & Furniture', 'Mud Terrain', 'Electric Scooters', 'Boots', 'Porch Swings & Hammocks', 'All Season', 'Table Lamps', 'Area Rugs', 'Active Bottoms', 'Storage Cabinets', 'Accent Pillows', 'Coffee Mugs & Tea Cups', 'Cleaning Tools', 'Dinnerware Sets', 'Outdoor Tables', 'Speakers & Subwoofers', 'Dress Shoes', 'Jackets', 'Camp Furniture', 'Sneakers', 'TVs', 'Tote Bags', 'T-Shirts & Tanks', 'Christmas', 'Shoulder Bags', 'End & Side Tables', 'Outdoor Furniture Sets', 'Crossbody Bags', 'Wired', 'All Terrain', 'Towers', 'Lawn Games', 'Bedding Sets', 'Cocktail Glasses & Tumblers', 'Bar & Counter Stools', 'Summer', 'Xbox', 'Chairs', 'Phones', 'Bluetooth Speakers', 'PS4 Games', 'Trash Cans & Recycling Bins', 'Recliners', 'Desks', 'Portable ACs', 'Curtains & Drapes', 'Sleeping Bags & Bedding', 'Hand Tools', 'Coats', 'Electric Fireplaces', 'Washer / Dryer Combinations', 'Pitchers', 'Ottomans & Poufs', 'Dining Chairs & Benches', 'Knife Sets', 'Nintendo', 'Camp Kitchen', 'Tents & Shelters', 'Sandals', 'Surveillance & Sensors', 'Coffee Grinders', 'Performance', "Men's Bracelets", 'Candles & Holders', 'Bath Towel Sets', 'Cutting Boards', 'Xbox One Games', 'Slippers', 'Automatic Drip', 'Bedframes & Boxsprings', 'Runners', 'Shirts', 'Food Storage', 'Tea Kettles', "Kids' Bikes & Scooters", 'Bookcases & Bookshelves', 'Backpacks', 'Hobo Bags', 'Baby Monitors', 'Desk Chairs', 'Room & Wall Accents', 'Snacks', 'Comforters', 'Wall Lights', 'Gloves', 'Bath Accessories', 'Living Room Sets', 'Nintendo Switch Games', 'Halloween', 'Baking Dishes', 'Tables', 'Sideboards & Buffets', 'Digital Cameras', 'Hats', 'Vests', 'Underwear & Boxers', 'Winter', 'Bikes', 'Top Load Washers', 'Couches & Sofas', 'Computer Accessories', 'Glassware & Drinkware', 'Cake & Pie Tins', 'Dishware', 'Cat Supplies & Care', 'Lanterns & Lighting', 'Highway', 'Windows PC Games', 'Outdoor Lights', 'Workout Machines', 'Kitchen Carts & Islands', 'Accent Table Sets', 'Xbox Series X Games', 'Bedroom Sets', 'Outdoor Chairs & Ottomans', 'Body Massagers', 'Belt Bags', 'Sheets & Pillow Cases', 'Adapters', 'Loafers & Clogs', 'Touring', 'Casual Pants', 'Smart Phones', 'Bath Rugs & Mats', 'Bedding & Pillows', 'Windows PC', 'TV Stands & Entertainment Centers', 'Wall Mounts', 'Tool Sets', 'Floor Lamps', 'Work Bags', 'Beds', 'Portable Fans', 'Sheet Pans', 'Water Bottles & Travel Mugs', 'Benches', 'Front Load Washers', 'Kitchen Knives', 'Power Tools', 'Casual Shirts', 'Cleaning', 'Pants', 'Cuff Links', 'Vanities & Storage', 'Pet Stain Stain & Odor Removers', 'Pendants', 'Turntables & Records', 'Pantry & Kitchen Cabinets', 'Beds & Crates', 'Hair Dryers', 'Storage', 'Toy Chests', 'Nightstands', 'Soundbars', 'Shampoo & Conditioner', 'Shoe Storage', 'All in One', 'Beverage Coolers', 'Xbox Series X/S Games', 'Through-the-Wall ACs', 'Wireless', "Men's Necklaces", 'Accent Furniture', 'Window ACs', 'Electric Ranges', 'Weights & Weight Sets', 'Accent Blankets', 'PS5', 'Displays & Storage', 'Bath Towels & Bath Sheets', 'Nintendo 3DS|2DS Games', 'Outdoor Dining Sets', 'Blocks & Play Sets', 'Yoga Mats & Workout Accessories', 'Bowls & Feeders', 'Tower Heaters', 'Binoculars & Telescopes', 'Phone Cases', 'Outdoor Couches & Benches', 'Retro', 'Components', 'Vases', 'PS5 Games', 'General', 'Wall Art', 'Headboards', 'Skill Building', 'Hall Trees', 'Mattress Only', 'Dining Tables', 'Espresso & Cappuccino Makers', 'Foam Rollers', 'Single Serve', 'Coffee Tables', "Kids' Tables & Chairs", 'French Door', 'Outdoor & Lawn Decor', 'Beverages', 'Ride On Toys', 'Trays & Baskets', 'Wash Cloths', 'Trampolines', 'Wall Mirrors', 'Toiletries', 'Single Serve Coffee Machines', 'Body Lotion & Oils', 'Hoverboards', 'Socks', 'Youth Bikes', 'Electric Dryers', 'Travel Accessories', 'Dressers & Armoires', 'Mattress Toppers', 'Weekenders', 'Heels', 'Quilts & Blankets', 'Gaming Laptops', 'Compact', 'Blinds & Shades', 'Shower Curtains', 'Swing Sets & Playhouses', 'Baby Care', 'Utensils & Tools', 'Flatware', 'Electric Wall Ovens', 'Hammocks', 'Faux Plants & Planters', 'Bags', 'Top Freezer', 'Kids Tablets', 'PlayStation', 'Hampers & Laundry Baskets', 'Shirts & Blouses', 'Dining Sets', 'Bowls', 'Storage & Accessories', 'Active Tops', 'Eye Shadow', 'Outdoor Accessories', 'Wine Coolers', 'Gas Dryers', 'Bed Pillows', 'Sofas', 'Scarves', 'Curling Irons', 'Helmuts, Tools & Accessories', 'Wall Clocks', 'Receivers', 'Attachments & Accessories', 'Accessories & Parts', 'Day Care', 'Beach Towels', 'Plates', 'Software', 'Portable Speakers', 'Grills, Griddles & Wafflers', 'Routers', 'Console Tables', 'Voice Assistants', 'Water & Juice Glasses', 'Upright Freezers', 'Masks', 'Dog Collars, Leashes, & Harnesses', 'Scooters', 'Amplifiers', 'Feminine Care', 'Crib & Toddler Mattresses', 'Filing Cabinets & Accessories', 'Foundation', 'Gas Ranges', 'Sound Systems', 'Bottles & Warmers', 'Dishes & Utensils', 'Wine & Champagne Glasses', 'Scanners', 'Health Monitors', 'Tower Fans', 'Whole Home Systems', 'Pedestals', 'Cribs', 'ATV/UTV', 'Subwoofers', 'Arm Chairs', 'PS4 VR Games', 'Highchairs & Boosters', 'Ethernet Switches', 'Belts', 'Smart Lighting', 'Metal Detectors', 'Powder', 'Cupcake & Muffin Tins', 'Floor Heaters', 'Dining Room Sets', 'Shelves & Hooks', 'PS4', 'Sleep', 'Countertop Microwaves', 'Toilet Paper', 'School & Office Supplies', 'Tool Storage & Organization', 'Air Tools & Compressors', 'Bar Tool Sets', 'Batteries', 'Hannukah', 'Equipment', 'Costumes & Dress Up', 'Radios/Scanners/Dash Cams', 'Side By Side', 'Deep Fryers', 'Patio Furniture', 'Hi Top Sets', 'Breast Pumps', 'Gear & Safety', 'Drying Racks', 'Dish Soap', "Men's Rings", 'Chest Freezers', 'Hair Straighteners', 'Pour Over', 'Dining Tables & Chairs', 'Go Karts', 'iPods', "Kids' Beds", 'Hand Soaps', 'Calculators', 'Cat Toys', 'Gas Cooktops', 'Ink Jet', 'French Press', 'Body Wash & Soap', "Kids' Desks", 'Changing Pads', 'Pre-Paid Phones', 'Cookware', 'Patio Sets & Chairs', 'Hangers', 'Lab Coats', 'Stacked Units', 'Clothing Racks', 'Detergent & Softener', '2 Piece Systems', 'Multi-use Tools', 'Spa & Relaxation', 'Litter Boxes', 'Hand Towels', 'Smart Thermostats', 'Outdoor Lamps & Lanterns', 'Shower & Tub Organizers', 'Wall Shelves', 'Kids Luggage', 'All-Purpose Cleaners', 'Disinfectant Wipes', 'Side Tables', 'Dressers & Mirrors', 'Conditioner', 'Island Hoods', 'Utensil Sets', 'Specialty', 'Camcorders', 'Bottom Freezer', 'Robotic Vacuums', 'Buffet & Storage', "Kids' Bookcases", 'Slow Cookers', 'Self Balancing Scooters', 'Paper Towels & Napkins', '5.1 Systems', 'Mac', 'Potties', '2.1 Systems', 'Pacifiers & Sleep Accessories', 'Mixing Glasses & Jiggers', 'Remote Starters', 'TV Stands', 'Rice Cookers & Steamers', 'Ice Makers', 'Floor Mirrors', 'Baby Gates', 'Clutches', 'Mascara', 'Outdoor Knives & Tools', 'Extenders', 'Under Counter Refrigerator', 'Sofas & Sofa Sets', 'Kegerators', 'Wine Openers', 'Reading', 'Indoor', 'Bath & Potty', 'Lightbulbs', 'Food Choppers', 'Polo Shirts', 'Mattress Pad Covers', 'Dress Pants', 'Tissues', 'Household', 'Outdoor Pillows & Cushions', 'Foundations', 'Changing Pad Covers', 'Smart Locks', 'Dog Crates', 'Cleanser', 'Diapers & Wipes', 'Outdoor Bars & Carts', 'Lounge Chairs', 'Arts & Crafts', 'Closet Systems', 'Baby Bags', 'Xbox 360 Games', 'Stick Vacuums', 'Warming Drawers', 'Easter', 'Bathroom Cleaners', 'Paper Plates & Cups', 'Baby Blankets', 'Electric Cooktops', 'Shakers & Stirrers', 'Trash Bins', 'Desktop Organization', 'Safety', 'Money Clips', 'Baby Bath Towels', 'Skateboards', 'Shorts', 'Mugs', 'Upright Vacuums', 'Kitchen Shears & Scissors', 'Treats', 'Eye Care', 'Floats', 'Night Care', 'Knife Storage', "Kids' Mattresses", 'Hair', 'Training', 'Throw Blankets', 'Accessory Bundles', 'Chaises', 'Cutlery', 'Bar Stools', 'Adult Bikes', 'Outdoor String Lights']

Output rules (important)

Output only the chosen label, nothing else (no punctuation, JSON, quotes, or explanations).

If multiple labels could apply, prefer the broadest/top-level option over granular ones unless the user explicitly uses an exact label string from the list.

Example: “I want a laptop” → Electronics (not “Laptops” or “Computers”).

Example: “Add to my ‘Women’s Clothing’ wishlist” → Women's Clothing (exact match requested).

Ignore brands, prices, emojis, and PII.

Handle synonyms and minor typos.

If nothing obviously fits, pick the closest label semantically; never invent labels and never output more than one.

Few-shot examples

User: “I want a laptop”
Assistant: Electronics

User: “iphone 15 case”
Assistant: Electronics

User: “looking for dumbbells and a yoga mat”
Assistant: Sports

User: “need a sectional sofa”
Assistant: Furniture

User: “wedding ring set”
Assistant: Wedding & Engagement

User: “dog food and treats”
Assistant: Pet

User: “nonstick pots and pans”
Assistant: Kitchen & Dining

User: “skin serum for acne”
Assistant: Beauty

User: “women’s running shoes”
Assistant: Shoes

User: “ps5 games”
Assistant: Video Games


User: {user_text}"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}],
            temperature=0.0,
        )
        category_guess = _strip_think(resp.choices[0].message.content).strip()
    except Exception:
        category_guess = ""

    return {
        "brand": "",
        "color": "",
        "object": "",
        "terms": [],
        "category_name_1": category_guess,
        "category_name_2": "",
        "category_name_3": ""
    }

# --- EXACT PROMPTS YOU REQUESTED -------------------------------------------
def extract_object(user_text: str) -> str:
    """
    Use the exact prompt to define a single object from the user's input.
    """
    prompt = f'Your job is to define the object given a user prompt. Only output one word describing the object.\nUser Prompt: "{user_text}"'
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        obj = _strip_think(resp.choices[0].message.content).strip()
        # keep it compact; single line; normalize a bit
        obj = re.sub(r"[\n\r]+", " ", obj).strip()
        return _canon_object(obj)
    except Exception:
        # fallback: grab a noun-ish token
        toks = re.findall(r"[a-zA-Z0-9\-]{3,}", user_text.lower())
        return _canon_object(toks[0]) if toks else ""

def extract_three_terms(obj: str) -> List[str]:
    """
    Use the exact prompt to get exactly three terms describing the object.
    """
    prompt = f'your job is to determine three terms to describe the object. Only output three words describing the object.\n\ngiven object: "{obj}"'
    cleaned: List[str] = []
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        raw = _strip_think(resp.choices[0].message.content).strip()
        # accept comma or newline separated outputs
        parts = re.split(r"[,\n]+", raw)
        cleaned = [_canon_object(p) for p in parts if p.strip()]
    except Exception:
        cleaned = []

    # ensure exactly three items (dedupe, trim/pad)
    uniq = []
    for t in cleaned:
        if t and t not in uniq:
            uniq.append(t)
    uniq = uniq[:3]
    while len(uniq) < 3:
        uniq.append(obj if obj and obj not in uniq else "")
    return [t for t in uniq if t]

def summarize_products(original_query: str, keywords: str, items: List[Dict]) -> str:
    simple = []
    for it in items[:5]:
        simple.append({k: it.get(k) for k in ["similarity","name","brand","current_price","product_url","product_id","score","business_score"] if k in it})
    system = (
        "You write concise, human-friendly summaries of product search results.\n"
        "Do NOT include analysis or <think> blocks. Output only the final text.\n"
        "Output should be a short intro + a 5-item bulleted list (name, brand, price if present), then a one-line suggestion.\n"
        "Use ONLY the provided products. Do not invent specs or prices."
    )
    user = json.dumps({"original_query": original_query, "keywords": keywords, "top5": simple}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return _strip_think(resp.choices[0].message.content.strip())
    except Exception:
        lines = [f"Top {len(simple)} matches for '{keywords}':"]
        for it in simple:
            nm = it.get("name") or "(name)"
            br = it.get("brand") or ""
            pr = it.get("current_price")
            line = f"- {nm}"
            if br: line += f" — {br}"
            if pr is not None: line += f" (${pr})"
            lines.append(line)
        return "\n".join(lines)

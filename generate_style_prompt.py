import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

style_prompts = {
    "style": [
        "business formal, suit and tie",
        "business casual, blazer with no tie",
        "smart casual, blazer and jeans",
        "minimalist style, monochrome, clean lines",
        "streetwear, oversized, graphic prints",
        "sporty, athleisure wear",
        "boho, ethnic patterns and loose fits",
        "vintage, retro style",
        "romantic, soft colors and ruffles",
        "edgy, avant-garde pieces",
        "normcore, neutral and plain clothing",
        "techwear, functional with many pockets"
    ],
    "color": [
        "neutral tones: white, black, beige",
        "cold tones: blue, grey, purple",
        "warm tones: red, yellow, orange",
        "pastel palette",
        "bright and saturated colors",
        "monochrome look"
    ],
    "cut_fit": [
        "cropped tops",
        "long loose coats",
        "fitted silhouettes",
        "oversized jackets",
        "high-waisted trousers",
        "flared pants",
        "straight cut"
    ],
    "era": [
        "modern style",
        "inspired by 90s fashion",
        "inspired by 80s fashion",
        "inspired by 70s fashion",
        "retro vibes from the 50s",
        "vintage grandma aesthetic",
        "japanese minimalist fashion",
        "french chic",
        "scandinavian style",
        "futuristic look"
    ],
    "brand_match": [
        "looks like COS or Uniqlo",
        "similar to Zara or Mango",
        "reminiscent of Off-White or Vetements",
        "could be Gucci or Prada",
        "similar to The Row or Celine",
        "like Ganni or Sandro",
        "classic Massimo Dutti vibes",
        "sporty like Nike or Adidas",
        "denim-heavy like Levi’s or Wrangler"
    ]
}

def detect_category(image_features, category_list):
    text_tokens = clip.tokenize(category_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    sims = cosine_similarity(image_features, text_features.cpu().numpy())[0]
    best_idx = np.argmax(sims)
    return category_list[best_idx]

def average_image_features(folder):
    features = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, filename)
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image)
        features.append(image_feature.cpu().numpy())
    return np.mean(features, axis=0)

def generate_prompt(image_folder):
    avg_feature = average_image_features(image_folder)
    result = []

    for category, descriptions in style_prompts.items():
        result.append(detect_category(avg_feature, descriptions))

    return "Style description:\n" + "\n".join(result)

if __name__ == "__main__":
    folder_path = "clothing_visible"
    prompt = generate_prompt(folder_path)
    print(prompt)

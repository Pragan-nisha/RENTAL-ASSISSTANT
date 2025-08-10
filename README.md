# RENTAL-ASSISSTANT
An AI-powered rental property search and analytics tool that delivers precise, hallucination-free insights from scraped data — no APIs required.


## ✨ Features

- **🔍 Smart Search** – Find rental properties by location, type (BHK), and size (sqft).
- **📊 Market Analytics** – Get average, minimum, maximum, and median rent instantly.
- **🚫 No Hallucinations** – If data isn’t in the database, it will politely say so.
- **📦 Offline & Fast** – Works entirely from your local ChromaDB.
- **🌍 Region-Locked Answers** – Ensures accuracy by restricting responses to available data.

---

## 🛠️ How It Works

1. **Scraping Engine** – Collects live rental listings from MagicBricks (price, address, amenities, BHK, bathrooms, kitchens, sqft).
2. **Vector Storage** – Stores structured data in **ChromaDB** with embeddings from **Qwen 2** (via Ollama).
3. **Query Understanding** – Uses **Gemini** to detect whether the request is a search or analytics query.
4. **Result Generation** – Returns only relevant, verified results — no made-up answers.

**Architecture:**
```

[MagicBricks Scraper] → [Structured Data] → [ChromaDB + Qwen2 Embeddings] → [Gemini Query Parser] → [Accurate Results]

````

---

## 📂 Tech Stack

| Layer            | Technology |
|------------------|------------|
| **Scraping**     | Python, BeautifulSoup/Selenium |
| **Database**     | ChromaDB |
| **Embeddings**   | Qwen 2 via Ollama |
| **Query Parsing**| Google Gemini 1.5 |
| **Geo Filtering**| geopy |
| **Language**     | Python 3.10+ |


---

## 🚀 Future Roadmap

* 📍 Multi-city scraping
* 🖥️ Interactive web dashboard
* 🎙️ Voice-based property queries

---

## 📌 Author

**Pragan Nisha K**
*Data & AI Enthusiast*

---

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

def clean_address(address: str) -> str:
    """
    Cleans the address by removing repeated segments.
    Example: 
        'Ramanathapuram, Coimbatore, Ramanathapuram, Coimbatore, Tamil Nadu'
        => 'Ramanathapuram, Coimbatore, Tamil Nadu'
    """
    parts = [p.strip() for p in address.split(',')]
    seen = set()
    cleaned_parts = []

    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            cleaned_parts.append(part)

    cleaned_address = ', '.join(cleaned_parts)
    return cleaned_address


# Global cache to store already geocoded addresses
address_cache = {}

def get_coordinates(address):
    """
    Geocode using LocationIQ only, with progressive fallback by shortening the address.
    Prevents repeated coordinates for similar addresses using a cache.
    """
    api_key = "LocationIQ API"
    url = "https://us1.locationiq.com/v1/search.php"
    headers = {
        "User-Agent": "GeoBot"
    }

    parts = [p.strip() for p in address.split(',')]
    tried = set()

    for i in range(len(parts)):
        try_address = ', '.join(parts[i:])
        if try_address.lower() in address_cache:
            lat, lon = address_cache[try_address.lower()]
            print(f"🔁 Cache hit: {try_address} → {lat}, {lon}")
            return lat, lon

        if try_address in tried:
            continue
        tried.add(try_address)

        params = {
            'key': api_key,
            'q': try_address,
            'format': 'json',
            'limit': 1,
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                print(f"LocationIQ match: {try_address} → {lat}, {lon}")
                # Save to cache (normalized lowercase address)
                address_cache[try_address.lower()] = (lat, lon)
                return lat, lon
        except Exception as e:
            print(f"Error geocoding '{try_address}': {e}")

        time.sleep(1)  # to prevent rate-limiting

    print("LocationIQ: No match after trying all fallbacks.")
    return None, None


# --- Function: Find nearby amenities ---
def find_nearby_places(lat, lon, radius=1500):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["amenity"];
      node(around:{radius},{lat},{lon})["shop"];
      node(around:{radius},{lat},{lon})["leisure"];
      node(around:{radius},{lat},{lon})["tourism"];
      node(around:{radius},{lat},{lon})["office"];
    );
    out center;
    """

    try:
        response = requests.post(overpass_url, data=query.encode("utf-8"))
        data = response.json()

        places = []
        for element in data["elements"]:
            if "tags" in element:
                name = element["tags"].get("name")
                if name:
                    places.append(name)
                else:
                    # fallback to type like 'school', 'hospital' etc.
                    places.append(element["tags"].get("amenity") or element["tags"].get("shop") or element["tags"].get("leisure") or element["tags"].get("tourism") or element["tags"].get("office"))
        return places[:5] # return top 5
    except Exception as e:
        print("Overpass error:", e)
        return []


def extract_property_details(driver,url):
    details = {}
    more_details = {}
    nearby_amenities = []
    title = None
    lat=0
    lon=0

    # --- TITLE ---
    try:
            r = requests.get(url, headers=HEADERS)
    
            if "Access Denied" in r.text:
                print("Access denied for:", url)
                return None

            soup = BeautifulSoup(r.text, "html.parser")

            # Extract data
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else None

    except:
        print("Title not found")

    # --- DETAILS section ---
    try:
        details_items = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "ul.mb-ldp__dtls__body__list li.mb-ldp__dtls__body__list--item")
            )
        )
        for li in details_items:
            label = li.find_element(By.CSS_SELECTOR, ".mb-ldp__dtls__body__list--label").text.strip()
            value = li.find_element(By.CSS_SELECTOR, ".mb-ldp__dtls__body__list--value").text.strip()
            details[label] = value
    except:
        print("Details section not found")

    # --- MORE DETAILS section ---
    try:
        more_items = driver.find_elements(By.CSS_SELECTOR, "ul.mb-ldp__more-dtl__list li.mb-ldp__more-dtl__list--item")
        for li in more_items:
            divs = li.find_elements(By.TAG_NAME, "div")
            if len(divs) >= 2:
                label = divs[0].text.strip()
                value = divs[1].text.strip()
                more_details[label] = value
    except:
        print("More details section not found")
        
    address = more_details.get('Address')
    if address:
        cleaned = clean_address(address)
        print(f"Cleaning address: {address} -> {cleaned}")
        lat, lon = get_coordinates(cleaned)
        if lat and lon:
            print(f"Coordinates: {lat}, {lon}")
            nearby_amenities = find_nearby_places(lat, lon)
            print(f"Amenities found: {nearby_amenities}")
        else:
            print("Geocoding failed.")
    else:
        print("No address found.")
    
    print (
        {"title": title,
        "details": details,
        "more_details": more_details,
        "amenities": nearby_amenities,
        "Lat":lat,
        "Lon":lon,
        "Link":url}
    )


    return {
        "title": title,
        "details": details,
        "more_details": more_details,
        "amenities": nearby_amenities,
        "Lat":lat,
        "Lon":lon,
        "Link":url
    }



# Headers for requests (not used in this version, but keep for BS4 if needed)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Setup driver
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)

city = "Coimbatore"
page = 1
property_urls = []
prop = []

while True:
    # Create pagination URL dynamically
    paginated_url = f"https://www.magicbricks.com/property-for-rent/residential-real-estate?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName={city}&page={page}"
    print(f"\n🌐 Loading page {page}: {paginated_url}")
    driver.get(paginated_url)

    # Handle CAPTCHA if any
    if 'captcha' in driver.title.lower() or 'blocked' in driver.title.lower():
        print("CAPTCHA detected. Please solve it manually, then press Enter...")
        input("Press Enter once CAPTCHA is cleared and listings are visible...")

    print(f"Page loaded: {driver.title}")

    # Wait for property cards to appear
    try:
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.mb-srp__card")))
        cards = driver.find_elements(By.CSS_SELECTOR, "div.mb-srp__card")
        print(f" Found {len(cards)} property cards on page {page}")
    except Exception as e:
        print(f"Could not find property cards: {e}")
        break

    if len(cards) == 0:
        print(" No more listings found. Ending loop.")
        break

    for i, card in enumerate(cards):
        try:
            print(f"\n🔹 Clicking property card {i+1}")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
            time.sleep(1)

            original_windows = driver.window_handles
            try:
                card.click()
            except:
                driver.execute_script("arguments[0].click();", card)

            time.sleep(3)
            new_windows = driver.window_handles

            if len(new_windows) > len(original_windows):
                driver.switch_to.window(new_windows[-1])
                current_url = driver.current_url
                print(f"  URL captured: {current_url}")
                data = extract_property_details(driver, current_url)
                prop.append(data)
                property_urls.append(current_url)
                driver.close()
                driver.switch_to.window(original_windows[0])
            else:
                current_url = driver.current_url
                print(f" URL captured (same tab): {current_url}")
                data = extract_property_details(driver, current_url)
                prop.append(data)
                property_urls.append(current_url)
                driver.back()
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.mb-srp__card")))
                time.sleep(2)

        except Exception as e:
            print(f"  Error processing card {i+1}: {e}")
            try:
                while len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue

    page += 1
    time.sleep(5)  # Optional delay to avoid IP blocks

driver.quit()

# Save to CSV
import json

# Save to JSON
if prop:
    filename = f"{city.lower()}_rentals.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prop, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(prop)} properties to '{filename}'")
else:
    print(" No property data was captured.")


import json
import os
import uuid
import time
from datetime import datetime
from typing import Dict, List, Tuple
import re
import requests

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

class PropertyDataLoader:
    def __init__(self, 
                 collection_name: str = "rental_properties",
                 db_path: str = "./chroma_db_rental_detailsS",
                 ollama_model: str = "qwen2.5:1.5b"):
        """Initialize data loader with Ollama only"""
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Test Ollama connection
        self._test_ollama_connection()

        # Init embeddings
        print("Loading embedding model...")
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded!")

        # Init Chroma
        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_function,
            collection_name=collection_name
        )
        print(f"Connected to Chroma database: {collection_name}")

    def _test_ollama_connection(self):
        """Test Ollama connection and model availability"""
        try:
            # Test basic connection
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
            print("Ollama server is running")
            
            # Test model availability
            response = requests.post(self.ollama_url, 
                json={
                    "model": self.ollama_model,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"Model {self.ollama_model} is available")
            else:
                raise Exception(f"Model {self.ollama_model} not available. Run: ollama pull {self.ollama_model}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure to run: ollama serve")
        except Exception as e:
            print(f"Ollama setup error: {e}")
            raise

    def extract_with_ollama(self, property_data: Dict) -> Dict:
        """Extract structured property info using Ollama"""
        
        title = property_data.get('title', '')
        details = property_data.get('details', {})
        more_details = property_data.get('more_details', {})
        amenities = property_data.get('amenities', [])
        lat = property_data.get('Lat', 0)
        lon = property_data.get('Lon', 0)
        
        prompt = f"""Extract property information and return ONLY valid JSON.

PROPERTY DATA:
Title: {title}
Details: {details}
More Details: {more_details}
Amenities: {amenities}
Latitude: {lat}
Longitude: {lon}

Return this EXACT JSON format (no markdown, no extra text):
{{
  "city": "extracted_city_name",
  "area": "extracted_area_name", 
  "property_type": "extracted_property_type",
  "monthly_price": extracted_price_as_number,
  "amenities": {json.dumps(amenities)},
  "lat": {lat},
  "lon": {lon}
}}

EXTRACTION RULES:
1. city: Find city name from title/address (like Coimbatore, Chennai)
2. area: Find locality/area name (like Kuniyamuthur, Peelamedu)
3. property_type: Find property type (like "3 BHK", "2 BHK House", "Villa")
4. monthly_price: Extract rent amount as number only (from Rental Value field)
5. amenities: Use provided amenities list **EXACTLY, NOT THE COUNT**.
6. lat/lon: Use provided coordinates as numbers

RESPOND WITH ONLY THE JSON OBJECT."""

        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2)  # Brief delay before retry
                
                response = requests.post(self.ollama_url, 
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 4096,
                            "num_predict": 300
                        }
                    },
                    timeout=60  # Longer timeout for Ollama
                )
                
                if response.status_code != 200:
                    print(f"Ollama API error: {response.status_code}")
                    continue
                
                response_data = response.json()
                response_text = response_data.get("response", "").strip()
                
                if not response_text:
                    print(f"Empty response from Ollama on attempt {attempt + 1}")
                    continue
                
                # Parse JSON from response
                parsed_data = self._parse_json_response(response_text)
                return parsed_data
                
            except requests.exceptions.Timeout:
                print(f"Ollama timeout on attempt {attempt + 1}")
                continue
            except Exception as e:
                print(f"Ollama error on attempt {attempt + 1}: {e}")
                continue
        
        # If all attempts failed, use fallback
        print("Ollama extraction failed, using manual fallback...")
        return self._fallback_extraction(property_data)

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from Ollama response"""
        try:
            # Clean the response aggressively
            response_text = response_text.strip()
            
            # Remove common markdown formatting
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Find JSON boundaries
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON structure found")
            
            json_str = response_text[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            # Validate and clean the parsed data
            result = {
                "city": str(parsed_data.get("city", "Coimbatore")).strip(),
                "area": str(parsed_data.get("area", "")).strip(),
                "property_type": str(parsed_data.get("property_type", "Property")).strip(),
                "monthly_price": int(parsed_data.get("monthly_price", 0)),
                "amenities": parsed_data.get("amenities", []) if isinstance(parsed_data.get("amenities"), list) else [],
                "lat": float(parsed_data.get("lat", 0)),
                "lon": float(parsed_data.get("lon", 0))
            }
            
            return result
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response_text[:200]}...")
            raise
        except Exception as e:
            print(f"Response parsing error: {e}")
            raise

    def _extract_city_fallback(self, property_data: Dict) -> str:
        """Extract city using fallback method"""
        title = property_data.get('title', '').lower()
        address = property_data.get('more_details', {}).get('Address', '').lower()
        
        # Common cities in Tamil Nadu
        cities = ['coimbatore', 'chennai', 'madurai', 'salem', 'trichy', 'erode', 'tirunelveli']
        
        for city in cities:
            if city in title or city in address:
                return city.title()
        
        return "Coimbatore"  # Default

    def _extract_area_fallback(self, property_data: Dict) -> str:
        """Extract area using fallback method"""
        title = property_data.get('title', '')
        address = property_data.get('more_details', {}).get('Address', '')
        
        # Extract area from title (usually after city name)
        title_match = re.search(r'Rent([^,]+)', title)
        if title_match:
            area = title_match.group(1).strip()
            # Remove city name if present
            area = re.sub(r'[,\s]*(coimbatore|chennai|madurai)', '', area, flags=re.IGNORECASE)
            if area:
                return area.strip()
        
        # Extract from address (first meaningful part)
        if address:
            parts = [p.strip() for p in address.split(',') if p.strip()]
            for part in parts[:3]:  # Check first 3 parts
                # Skip numbers and common words
                if not re.match(r'^\d+$', part) and len(part) > 2:
                    return part
        
        return ""

    def _extract_type_fallback(self, property_data: Dict) -> str:
        """Extract property type using fallback method"""
        title = property_data.get('title', '')
        
        # Extract BHK pattern
        bhk_match = re.search(r'(\d+\s*BHK)', title, re.IGNORECASE)
        if bhk_match:
            bhk = bhk_match.group(1)
            # Add "House" or "Apartment" if available
            if 'house' in title.lower():
                return f"{bhk} House"
            elif any(word in title.lower() for word in ['apartment', 'flat']):
                return f"{bhk} Apartment"
            else:
                return bhk
        
        # Look for other property types
        property_types = ['house', 'flat', 'apartment', 'villa', 'penthouse', 'independent']
        title_lower = title.lower()
        
        for prop_type in property_types:
            if prop_type in title_lower:
                return prop_type.title()
        
        return "Property"

    def _extract_price_fallback(self, property_data: Dict) -> int:
        """Extract price using fallback method"""
        rental_value = property_data.get('more_details', {}).get('Rental Value', '')
        
        if rental_value:
            # Extract number from "₹30,000" format
            price_match = re.search(r'₹([0-9,]+)', rental_value)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                try:
                    return int(price_str)
                except ValueError:
                    pass
        
        return 0

    def _fallback_extraction(self, property_data: Dict) -> Dict:
        """Manual extraction when Ollama fails - includes lat/lon"""
        return {
            "city": self._extract_city_fallback(property_data),
            "area": self._extract_area_fallback(property_data),
            "property_type": self._extract_type_fallback(property_data),
            "monthly_price": self._extract_price_fallback(property_data),
            "amenities": property_data.get('amenities', []) if isinstance(property_data.get('amenities'), list) else [],
            "lat": float(property_data.get('Lat', 0)),
            "lon": float(property_data.get('Lon', 0))
        }

    def create_property_document(self, property_data: Dict) -> Tuple[str, Dict]:
        """Create vector-searchable doc using Ollama extraction or fallback"""
        try:
            # Try Ollama first
            parsed = self.extract_with_ollama(property_data)
        except Exception as e:
            print(f"Ollama extraction failed: {e}")
            print(" Using manual fallback...")
            parsed = self._fallback_extraction(property_data)
        
        title = property_data.get('title', '')
        details = property_data.get('details', {})
        more_details = property_data.get('more_details', {})

        # Build document text for embedding
        document_text = f"""
        Property Title: {title}
        Property Type: {parsed['property_type']}
        Location: {parsed['area']}, {parsed['city']}
        Price: ₹{parsed['monthly_price']:,} per month
        Coordinates: {parsed['lat']}, {parsed['lon']}
        Details: {json.dumps(details, ensure_ascii=False)}
        More Details: {json.dumps(more_details, ensure_ascii=False)}
        Amenities: {', '.join(parsed['amenities'])}
        """.strip()

        # Metadata for filtering and retrieval - INCLUDES LAT/LON
        metadata = {
            "title": title,
            "city": parsed['city'].lower(),
            "area": parsed['area'].lower(),
            "property_type": parsed['property_type'],
            "price": parsed['monthly_price'],
            "lat": parsed['lat'],  #  Lat stored in metadata
            "lon": parsed['lon'],  #  Lon stored in metadata
            "link": property_data.get('Link', ''),
            "details": json.dumps(details, ensure_ascii=False),
            "more_details": json.dumps(more_details, ensure_ascii=False),
            "amenities": json.dumps(parsed['amenities'], ensure_ascii=False),
            "added_at": datetime.now().isoformat(),
            "property_id": str(uuid.uuid4())
        }

        return document_text, metadata
    
    def add_property(self, property_data: Dict) -> str:
        """Add single property to vector database using Ollama"""
        try:
            # Create document and metadata
            document_text, metadata = self.create_property_document(property_data)
            
            # Create LangChain Document
            document = Document(
                page_content=document_text,
                metadata=metadata
            )
            
            # Add to Chroma using LangChain
            ids = self.db.add_documents([document])
            
            return metadata["property_id"]
            
        except Exception as e:
            print(f"Error adding property: {e}")
            return None
    
    def load_from_json_file(self, json_file_path: str) -> int:
        """Load properties from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Handle different JSON structures
            if isinstance(data, list):
                properties = data
            elif isinstance(data, dict) and 'properties' in data:
                properties = data['properties']
            else:
                properties = [data]  # Single property
            
            return self.load_from_json_list(properties)
            
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return 0
    
    def load_from_json_list(self, properties_list: List[Dict]) -> int:
        """Load properties from Python list of dictionaries using batch processing"""
        try:
            documents = []
            added_count = 0
            batch_size = 3  # Small batches for Ollama to avoid timeouts
            
            # Process in batches
            for batch_start in range(0, len(properties_list), batch_size):
                batch_end = min(batch_start + batch_size, len(properties_list))
                batch = properties_list[batch_start:batch_end]
                
                print(f"\n Processing batch {batch_start//batch_size + 1}/{(len(properties_list)-1)//batch_size + 1}")
                
                # Prepare documents in current batch
                for i, property_data in enumerate(batch):
                    global_index = batch_start + i + 1
                    try:
                        document_text, metadata = self.create_property_document(property_data)
                        
                        document = Document(
                            page_content=document_text,
                            metadata=metadata
                        )
                        documents.append(document)
                        
                        # Verify lat/lon are included
                        lat_lon_info = f"(Lat: {metadata['lat']:.3f}, Lon: {metadata['lon']:.3f})"
                        print(f"Prepared property {global_index}/{len(properties_list)}: {property_data.get('title', 'Unknown')[:40]}... {lat_lon_info}")
                        
                        # Small delay between properties
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error preparing property {global_index}: {e}")
                        # Still try to add with fallback
                        try:
                            fallback_data = self._fallback_extraction(property_data)
                            title = property_data.get('title', 'Unknown Property')
                            
                            document_text = f"Property: {title}, Type: {fallback_data['property_type']}, Location: {fallback_data['area']}, {fallback_data['city']}, Price: ₹{fallback_data['monthly_price']}"
                            
                            metadata = {
                                "title": title,
                                "city": fallback_data['city'].lower(),
                                "area": fallback_data['area'].lower(),
                                "property_type": fallback_data['property_type'],
                                "price": fallback_data['monthly_price'],
                                "lat": fallback_data['lat'],
                                "lon": fallback_data['lon'],
                                "link": property_data.get('Link', ''),
                                "amenities": json.dumps(fallback_data['amenities']),
                                "added_at": datetime.now().isoformat(),
                                "property_id": str(uuid.uuid4())
                            }
                            
                            document = Document(page_content=document_text, metadata=metadata)
                            documents.append(document)
                            print(f"Added property {global_index} using fallback method")
                            
                        except Exception as fe:
                            print(f"Fallback also failed for property {global_index}: {fe}")
                
                # Delay between batches to avoid overwhelming Ollama
                if batch_end < len(properties_list):
                    print("⏳ Waiting 3s between batches...")
                    time.sleep(3)
            
            # Batch add all documents to Chroma
            if documents:
                print(f"\n🚀 Adding {len(documents)} documents to vector database...")
                self.db.add_documents(documents)
                added_count = len(documents)
                
                # Persist the database
                self.db.persist()
                print(f"Persisted database to disk")
            
            print(f"\nSuccessfully loaded {added_count}/{len(properties_list)} properties with lat/lon coordinates!")
            return added_count
            
        except Exception as e:
            print(f"Error in batch loading: {e}")
            return 0
    
    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            # Delete the entire directory and recreate
            import shutil
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            
            # Recreate the database
            self.db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection including lat/lon info"""
        try:
            # Use LangChain's similarity search to get sample data
            sample_docs = self.db.similarity_search("property", k=200)
            
            total_count = len(sample_docs)
            cities = set()
            property_types = set()
            prices = []
            has_coordinates = 0
            
            for doc in sample_docs:
                metadata = doc.metadata
                if metadata.get('city'):
                    cities.add(metadata['city'].title())
                if metadata.get('property_type'):
                    property_types.add(metadata['property_type'])
                if metadata.get('price', 0) > 0:
                    prices.append(metadata['price'])
                if metadata.get('lat', 0) != 0 and metadata.get('lon', 0) != 0:
                    has_coordinates += 1
            
            stats = {
                "total_properties": total_count,
                "properties_with_coordinates": has_coordinates,
                "coordinate_coverage": f"{(has_coordinates/total_count*100):.1f}%" if total_count > 0 else "0%",
                "cities": list(sorted(cities)),
                "property_types": list(sorted(property_types)),
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "avg": sum(prices) // len(prices) if prices else 0
                },
                "collection_name": self.collection_name,
                "database_path": self.db_path
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "total_properties": 0,
                "properties_with_coordinates": 0,
                "coordinate_coverage": "0%",
                "cities": [],
                "property_types": [],
                "price_range": {"min": 0, "max": 0, "avg": 0},
                "collection_name": self.collection_name,
                "database_path": self.db_path
            }
    
    def search_properties(self, query: str, k: int = 5) -> List[Dict]:
        """Search properties and return results with lat/lon"""
        try:
            docs = self.db.similarity_search(query, k=k)
            results = []
            
            for doc in docs:
                result = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "coordinates": {
                        "lat": doc.metadata.get('lat', 0),
                        "lon": doc.metadata.get('lon', 0)
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []

    def search_by_location(self, lat: float, lon: float, radius_km: float = 5, k: int = 10) -> List[Dict]:
        """Search properties by geographic proximity (requires lat/lon in metadata)"""
        try:
            import math
            
            # Get all properties first
            all_docs = self.db.similarity_search("property", k=1000)
            
            nearby_properties = []
            
            for doc in all_docs:
                metadata = doc.metadata
                prop_lat = metadata.get('lat', 0)
                prop_lon = metadata.get('lon', 0)
                
                if prop_lat != 0 and prop_lon != 0:
                    # Calculate distance using Haversine formula
                    distance = self._calculate_distance(lat, lon, prop_lat, prop_lon)
                    
                    if distance <= radius_km:
                        result = {
                            "content": doc.page_content[:200] + "...",
                            "metadata": metadata,
                            "distance_km": round(distance, 2),
                            "coordinates": {"lat": prop_lat, "lon": prop_lon}
                        }
                        nearby_properties.append(result)
            
            # Sort by distance
            nearby_properties.sort(key=lambda x: x['distance_km'])
            
            return nearby_properties[:k]
            
        except Exception as e:
            print(f"Error in location search: {e}")
            return []

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r

def main():
    """Main function to run with Ollama"""
    print("Starting Property Data Loader with Ollama...")
    print("=" * 60)
    
    try:
        # Initialize with Ollama only
        loader = PropertyDataLoader(ollama_model="qwen2.5:1.5b")
        
        # Test with single property first
        print("\n Testing with single property...")
        test_property = {
            "title": "3 BHK Residential House For RentKuniyamuthur, Coimbatore",
            "details": {
                "Carpet Area": "2000\nsqft\n₹15/sqft",
                "Status": "Immediately",
                "Facing": "East",
                "Furnished Status": "Semi-Furnished",
                "Age Of Construction": "10 to 15 years"
            },
            "more_details": {
                "Rental Value": "₹30,000",
                "Address": "16, Annai Illam, Kumaran Garden, Kuniyamuthur, Coimbatore, Kuniyamuthur, Coimbatore, Tamil Nadu",
                "Furnishing": "Semi-Furnished",
                "Age of Construction": "10 to 15 years",
                "Additional Rooms": "Puja Room"
            },
            "amenities": [
                "Krishna Engineering College",
                "sri krishna arts and science college",
                "IBP",
                "Vasan eye Care Hospital",
                "Sri Krishna College of Engineering and Technology"
            ],
            "Lat": 10.950426,
            "Lon": 76.955215,
            "Link": "https://www.magicbricks.com/propertyDetails/3-BHK-2000-Sq-ft-Residential-House-FOR-Rent-Kuniyamuthur-in-Coimbatore&id=4d423739313239383739"
        }
        
        property_id = loader.add_property(test_property)
        if property_id:
            print(f"Test property added: {property_id}")
            
            # Verify lat/lon storage
            results = loader.search_properties("3 BHK Kuniyamuthur", k=1)
            if results:
                meta = results[0]['metadata']
                print(f"Lat/Lon verified: {meta['lat']}, {meta['lon']}")
        
        # Load full dataset
        json_file_path = r"C:\Users\Pragan Nisha\New folder\coimbatore_rentals(backuppp).json"
        print(f"\nLoading full dataset from: {json_file_path}")
        added_count = loader.load_from_json_file(json_file_path)
        
        # Display statistics
        print("\n Collection Statistics:")
        stats = loader.get_collection_stats()
        print(f"   Total Properties: {stats['total_properties']}")
        print(f"   With Coordinates: {stats['properties_with_coordinates']} ({stats['coordinate_coverage']})")
        print(f"   Cities: {', '.join(stats['cities'])}")
        print(f"   Property Types: {', '.join(stats['property_types'][:5])}...")
        print(f"   Price Range: ₹{stats['price_range']['min']:,} - ₹{stats['price_range']['max']:,}")
        print(f"   Average Price: ₹{stats['price_range']['avg']:,}")
        
        # Test location-based search
        print(f"\nTesting location-based search...")
        nearby = loader.search_by_location(10.950426, 76.955215, radius_km=2, k=3)
        print(f"Found {len(nearby)} properties within 2km of test location:")
        for i, prop in enumerate(nearby, 1):
            meta = prop['metadata']
            print(f"   {i}. {meta['title'][:40]}... - {prop['distance_km']}km away")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Install the model: ollama pull qwen2.5:1.5b")
        print("   3. Test connection: curl http://localhost:11434/api/version")

if __name__ == "__main__":
    main()
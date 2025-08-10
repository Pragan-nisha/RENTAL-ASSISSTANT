"""
Conversational Property Insights System
--------------------------------------

This wrapper converts the JSON responses from the property insights system into
natural, conversational responses with memory support for continued conversations.

DEPENDENCIES:
- All dependencies from the original system
- langchain memory components: pip install langchain

"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from statistics import mean, median
from geopy.distance import geodesic

# genai SDK import
import google.generativeai as genai

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


class OfflinePropertyInsightsSystem:
    def __init__(
        self,
        chroma_persist_path: str,
        chroma_collection_name: str = "rental_properties",
        gemini_key_search: Optional[str] = None,
        gemini_key_analytics: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        # Keys
        self.gemini_key_search = gemini_key_search
        self.gemini_key_analytics = gemini_key_analytics 

        # Initialize Chroma and embeddings
        print("🔄 Initializing embeddings and ChromaDB connection...")
        self.embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.db = Chroma(
            persist_directory=chroma_persist_path,
            embedding_function=self.embedding_function,
            collection_name=chroma_collection_name,
        )

        # Load basic location index from DB metadata to enforce no-hallucination
        self._load_location_index()

        # Gemini model name
        self.gemini_model_name = "gemini-1.5-flash"

        

    def _call_gemini(self, prompt: str, intent: str = "search", max_tokens: int = 512) -> str:
        key = self.gemini_key_search if intent == "search" else self.gemini_key_analytics
        if not key:
            raise RuntimeError(f"No Gemini API key configured for intent '{intent}'.")

        genai.configure(api_key=key)
        model = genai.GenerativeModel(self.gemini_model_name)
        resp = model.generate_content(prompt)
        return resp.text

    def parse_query(self, user_query: str, intent_hint: Optional[str] = None) -> Dict:
        prompt = f"""
You are a JSON extractor for property database queries. Return ONLY a JSON object.
EXTRACT THE INTENT OF THE USER FOR EXAMPLE: If the question is about comparing prices, rents, averages, trends, or statistics between locations, set "intent" = "analytics" and fill the "comparison" field accordingly.

Schema:
{{
  "intent": "search|analytics",
  "locations": ["list of locations or landmarks"],
  "radius_km": null or number,
  "property_type": "apartment|house|pg|" or empty,
  "bhk": null or integer,
  "min_price": null or number,
  "max_price": null or number,
  "amenities": ["list"],
  "metric": "avg|min|max|median|count|trend|" or empty,
  "comparison": null or {{"op":"lt|gt|eq","locations":["A","B"]}},
  "time_range": null or {{"from":"YYYY-MM-DD","to":"YYYY-MM-DD"}},
  "raw_query": {json.dumps(user_query)}
}}
If unsure about a value, use null or empty lists. Return only the JSON blob.
"""
        if intent_hint:
            prompt = f"(PREFER INTENT={intent_hint})\n" + prompt

        text = self._call_gemini(prompt, intent=(intent_hint or "search"))
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Failed to extract JSON from LLM response")
        parsed = json.loads(match.group(0))
        # Normalize keys
        parsed.setdefault("intent", "search")
        parsed.setdefault("locations", [])
        parsed.setdefault("radius_km", None)
        parsed.setdefault("amenities", [])
        parsed.setdefault("metric", None)
        return parsed

    def _load_location_index(self):
        self.available_locations = set()
        self._docs_sample = []
        try:
            docs = self.db.similarity_search("__list_all_locations__", k=2000)
            for doc in docs:
                md = doc.metadata
                for field in (md.get('city', ''), md.get('area', ''), md.get('title', '')):
                    if field:
                        self.available_locations.add(str(field).strip().lower())
                self._docs_sample.append({'metadata': md, 'page_content': doc.page_content})
        except Exception:
            pass

    def resolve_location_from_db(self, label: str) -> Optional[Tuple[float, float, str]]:
        if not label:
            return None
        label_norm = label.strip().lower()
        if label_norm in self.available_locations:
            for entry in self._docs_sample:
                md = entry['metadata']
                city = str(md.get('city', '')).strip().lower()
                area = str(md.get('area', '')).strip().lower()
                title = str(md.get('title', '')).strip().lower()
                lat = md.get('lat')
                lon = md.get('lon')
                if lat is not None and lon is not None and (label_norm == city or label_norm == area or label_norm in title):
                    return (float(lat), float(lon), label_norm)
        for entry in self._docs_sample:
            md = entry['metadata']
            city = str(md.get('city', '')).strip().lower()
            area = str(md.get('area', '')).strip().lower()
            title = str(md.get('title', '')).strip().lower()
            lat = md.get('lat')
            lon = md.get('lon')
            if lat is None or lon is None:
                continue
            if label_norm in city or label_norm in area or label_norm in title:
                return (float(lat), float(lon), label_norm)
        return None

    def _filter_properties(self, filters: Dict) -> List[Dict]:
        prop_type = filters.get('property_type') or ''
        bhk = filters.get('bhk')
        min_price = filters.get('min_price')
        max_price = filters.get('max_price')
        amenities = [a.lower() for a in (filters.get('amenities') or [])]

        center = None
        radius_km = filters.get('radius_km')
        if filters.get('locations'):
            resolved = self.resolve_location_from_db(filters['locations'][0])
            if resolved:
                center = (resolved[0], resolved[1])
                if radius_km is None:
                    radius_km = 3.0

        raw_search_terms = ' '.join(filters.get('locations', []) or []) + ' ' + (filters.get('property_type') or '')
        try:
            candidates = self.db.similarity_search(raw_search_terms.strip() or "all_properties", k=1000)
        except Exception:
            candidates = []

        results = []
        for doc in candidates:
            md = doc.metadata
            try:
                price = int(md.get('price', 0))
            except Exception:
                price = 0

            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price != 0 and price > max_price:
                continue

            if prop_type:
                if prop_type.lower() not in str(md.get('property_type', '')).lower():
                    continue

            if bhk is not None:
                bhk_meta = md.get('bhk') or md.get('bedrooms') or ''
                try:
                    if int(bhk_meta) != int(bhk):
                        continue
                except Exception:
                    pass

            amen_meta = md.get('amenities', [])
            if isinstance(amen_meta, str):
                try:
                    amen_meta = json.loads(amen_meta)
                except Exception:
                    amen_meta = [a.strip() for a in amen_meta.split(',') if a.strip()]
            amen_meta_l = [str(x).lower() for x in (amen_meta or [])]
            if amenities:
                if not all(any(a in am for am in amen_meta_l) for a in amenities):
                    continue

            lat = md.get('lat')
            lon = md.get('lon')
            if center:
                if lat is None or lon is None:
                    continue
                distance_km = geodesic(center, (float(lat), float(lon))).km
                if radius_km is not None and distance_km > float(radius_km):
                    continue
            else:
                distance_km = None

            prop = {
                'title': md.get('title') or doc.page_content[:80],
                'city': md.get('city', ''),
                'area': md.get('area', ''),
                'lat': float(lat) if lat is not None else None,
                'lon': float(lon) if lon is not None else None,
                'price': price,
                'property_type': md.get('property_type', ''),
                'amenities': amen_meta_l,
                'url': md.get('url') or md.get('link') or '',
                'distance_km': round(distance_km, 2) if distance_km is not None else None,
                'raw_metadata': md,
            }
            results.append(prop)

        results.sort(key=lambda x: ((x['distance_km'] if x['distance_km'] is not None else 9999), x['price'] if x['price'] else 99999999))
        return results

    def compute_basic_stats(self, properties: List[Dict]) -> Dict:
        prices = [p['price'] for p in properties if p['price'] and p['price'] > 0]
        if not prices:
            return {'count': len(properties), 'has_price': False}
        return {
            'count': len(properties),
            'has_price': True,
            'min': min(prices),
            'max': max(prices),
            'avg': int(mean(prices)),
            'median': int(median(prices)),
        }

    def analytics_intent(self, parsed: Dict) -> Dict:
        locations = parsed.get('locations') or []
        metric = parsed.get('metric') or 'avg'
        comparison = parsed.get('comparison')

        location_results = {}
        for loc in locations:
            local_filters = parsed.copy()
            local_filters['locations'] = [loc]
            props = self._filter_properties(local_filters)
            if not props:
                location_results[loc] = {'status': 'no_data', 'properties': []}
            else:
                location_results[loc] = {'status': 'ok', 'properties': props, 'stats': self.compute_basic_stats(props)}

        if comparison and isinstance(comparison, dict):
            locs = comparison.get('locations') or []
            if len(locs) < 2:
                return {'status': 'error', 'message': 'Comparison requires two locations.'}
            lhs, rhs = locs[0], locs[1]
            lhs_res = location_results.get(lhs)
            rhs_res = location_results.get(rhs)
            if not lhs_res or lhs_res.get('status') == 'no_data':
                return {'status': 'no_data', 'message': f'No data available for {lhs}'}
            if not rhs_res or rhs_res.get('status') == 'no_data':
                return {'status': 'no_data', 'message': f'No data available for {rhs}'}

            lhs_val = lhs_res['stats'].get(metric)
            rhs_val = rhs_res['stats'].get(metric)
            if lhs_val is None or rhs_val is None:
                return {'status': 'error', 'message': 'Metric not available for comparison.'}

            op = comparison.get('op', 'lt')
            if op == 'lt':
                result_bool = lhs_val < rhs_val
            elif op == 'gt':
                result_bool = lhs_val > rhs_val
            else:
                result_bool = lhs_val == rhs_val

            return {
                'status': 'ok',
                'type': 'comparison',
                'metric': metric,
                'lhs': {'location': lhs, 'value': lhs_val},
                'rhs': {'location': rhs, 'value': rhs_val},
                'result': result_bool,
                'difference': abs(lhs_val - rhs_val)
            }

        if len(locations) == 1:
            loc = locations[0]
            res = location_results.get(loc)
            if not res or res.get('status') == 'no_data':
                return {'status': 'no_data', 'message': f'No data available for {loc}'}
            return {
                'status': 'ok',
                'type': 'stats',
                'location': loc,
                'stats': res['stats'],
                'sample_properties': [{'title': p['title'], 'price': p['price'], 'url': p['url']} for p in res['properties'][:10]]
            }

        return {'status': 'ok', 'locations': location_results}

    def search_intent(self, parsed: Dict) -> Dict:
        props = self._filter_properties(parsed)
        if not props:
            return {'status': 'no_data', 'message': 'No data available for the specified location/criteria.'}
        return {'status': 'ok', 'count': len(props), 'results': props[:50]}

    def handle_query(self, user_query: str, intent_hint: Optional[str] = None) -> Dict:
        parsed = self.parse_query(user_query, intent_hint=intent_hint or 'search')

        unresolved = []
        for loc in parsed.get('locations', []):
            resolved = self.resolve_location_from_db(loc)
            if not resolved:
                unresolved.append(loc)
        if unresolved:
            return {'status': 'no_data', 'message': f'No data available for location(s): {unresolved}'}

        if parsed.get('intent') == 'analytics':
            return self.analytics_intent(parsed)
        else:
            return self.search_intent(parsed)


class ConversationalPropertySystem:
    def __init__(
        self,
        chroma_persist_path: str,
        chroma_collection_name: str = "rental_properties",
        gemini_key_search: Optional[str] = None,
        gemini_key_analytics: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        memory_window: int = 10
    ):
        # Initialize the core property system
        self.property_system = OfflinePropertyInsightsSystem(
            chroma_persist_path=chroma_persist_path,
            chroma_collection_name=chroma_collection_name,
            gemini_key_search=gemini_key_search,
            gemini_key_analytics=gemini_key_analytics,
            embedding_model=embedding_model
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Store last search results for follow-up questions
        self.last_results = None
        self.last_query_context = None
        
        print("💬 Conversational property system ready!")

    def _format_search_response(self, result: Dict, user_query: str) -> str:
        """Convert search JSON response to natural language"""
        if result['status'] == 'no_data':
            # Clear any previous results since this search failed
            self.last_results = None
            self.last_query_context = None
            return f"I couldn't find any properties matching your criteria. {result.get('message', '')} Would you like to try searching in a different area or adjust your requirements?"
        
        properties = result['results']
        count = result['count']
        
        # Store results for potential follow-up ONLY if we have actual results
        self.last_results = properties
        self.last_query_context = user_query
        
        response = f"I found {count} properties matching your search"
        if count > 50:
            response += " (showing top 50 results)"
        response += ":\n\n"
        
        # Show top 5-10 results with details
        display_count = min(10, len(properties))
        for i, prop in enumerate(properties[:display_count], 1):
            response += f"**{i}. {prop['title']}**\n"
            response += f"📍 {prop['area']}, {prop['city']}\n"
            
            if prop['price']:
                response += f"💰 ₹{prop['price']}/month\n"
            
            if prop['property_type']:
                response += f"🏠 {prop['property_type'].title()}\n"
            
            if prop['distance_km']:
                response += f"📏 {prop['distance_km']} km from center\n"
            
            if prop['amenities']:
                amenities_str = ", ".join(prop['amenities'][:3])
                if len(prop['amenities']) > 3:
                    amenities_str += f" (+{len(prop['amenities'])-3} more)"
                response += f"✨ {amenities_str}\n"
            
            if prop['url']:
                response += f"🔗 [View Property]({prop['url']})\n"
            
            response += "\n"
        
        if count > display_count:
            response += f"...and {count - display_count} more properties available.\n\n"
        
        response += "Would you like me to show more details, filter these results further, or search for something else?"
        
        return response

    def _format_analytics_response(self, result: Dict, user_query: str) -> str:
        """Convert analytics JSON response to natural language"""
        if result['status'] == 'no_data':
            return f"I don't have enough data to provide analytics for your query. {result.get('message', '')} Try asking about a different location."
        
        if result['status'] == 'error':
            return f"I encountered an issue: {result.get('message', '')} Could you rephrase your question?"
        
        response_type = result.get('type', 'stats')
        
        if response_type == 'comparison':
            lhs = result['lhs']
            rhs = result['rhs']
            metric = result['metric']
            difference = result['difference']
            
            metric_name = {
                'avg': 'average rent',
                'min': 'minimum rent',
                'max': 'maximum rent',
                'median': 'median rent',
                'count': 'number of properties'
            }.get(metric, metric)
            
            response = f"Comparing {metric_name} between {lhs['location']} and {rhs['location']}:\n\n"
            response += f"📊 **{lhs['location'].title()}**: ₹{lhs['value']} {'per month' if 'rent' in metric_name else ''}\n"
            response += f"📊 **{rhs['location'].title()}**: ₹{rhs['value']} {'per month' if 'rent' in metric_name else ''}\n\n"
            
            if result['result']:
                response += f"✅ {lhs['location'].title()} has {'lower' if 'lt' in str(result) else 'higher'} {metric_name} than {rhs['location'].title()}"
            else:
                response += f"❌ {lhs['location'].title()} does not have {'lower' if 'lt' in str(result) else 'higher'} {metric_name} than {rhs['location'].title()}"
            
            if 'rent' in metric_name or 'price' in metric_name:
                response += f"\n💰 Difference: ₹{difference}"
            
        elif response_type == 'stats':
            location = result['location']
            stats = result['stats']
            samples = result.get('sample_properties', [])
            
            response = f"📈 **Rental Statistics for {location.title()}**\n\n"
            
            if stats['has_price']:
                response += f"🏠 Properties found: {stats['count']}\n"
                response += f"💰 Average rent: ₹{stats['avg']}\n"
                response += f"💰 Median rent: ₹{stats['median']}\n"
                response += f"💰 Price range: ₹{stats['min']} - ₹{stats['max']}\n\n"
                
                if samples:
                    response += "**Sample Properties:**\n"
                    for i, prop in enumerate(samples[:5], 1):
                        response += f"{i}. {prop['title']} - ₹{prop['price']}/month"
                        if prop['url']:
                            response += f" [View]({prop['url']})"
                        response += "\n"
            else:
                response += f"Found {stats['count']} properties, but pricing information is not available.\n"
        
        else:
            response = "📊 **Multi-location Analysis**\n\n"
            locations = result.get('locations', {})
            for loc, data in locations.items():
                response += f"**{loc.title()}:**\n"
                if data['status'] == 'ok':
                    stats = data['stats']
                    if stats['has_price']:
                        response += f"  • {stats['count']} properties, avg: ₹{stats['avg']}\n"
                    else:
                        response += f"  • {stats['count']} properties (no pricing data)\n"
                else:
                    response += f"  • No data available\n"
                response += "\n"
        
        return response

    def _check_followup_context(self, user_query: str) -> bool:
        """Check if the query might be a follow-up to previous results"""
        followup_indicators = [
            'show more', 'more details', 'first one', 'second one', 'that property',
            'refine', 'filter', 'narrow down', 'within', 'closer', 'cheaper',
            'expensive', 'larger', 'smaller', 'with', 'without', 'lesser', 'less than',
            'lower', 'higher', 'above', 'below'
        ]
        
        query_lower = user_query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)

    def _enhance_query_with_context(self, user_query: str) -> str:
        """Enhance follow-up queries with context from previous search"""
        # Only use context if we have actual results from previous search
        if (not self.last_query_context or 
            not self.last_results or 
            not self._check_followup_context(user_query)):
            return user_query
        
        # Add context from previous query
        enhanced = f"Previous search: {self.last_query_context}. Current request: {user_query}"
        return enhanced

    def chat(self, user_query: str) -> str:
        try:
            if self._is_followup_request(user_query) and self.last_query:
                merged_query = self._merge_queries(self.last_query, user_query)
                enhanced_query = self._enhance_query_with_context(merged_query)
            else:
                enhanced_query = self._enhance_query_with_context(user_query)
                self.last_query = user_query

            # Pass the query to property system to get structured result
            result = self.property_system.handle_query(enhanced_query)

            # Step 2: Save results
            if 'results' in result:
                self.last_results = result['results']

            # Step 3: Format response
            if result.get('intent') == 'analytics':
                return self._format_analytics_response(result, user_query)
            elif 'results' in result:
                return self._format_search_response(result, user_query)
            else:
                return "I couldn't match that request to a property search. Could you clarify?"

        except Exception as e:
            return f"Error processing your request: {e}"


    def _is_followup_request(self, text: str) -> bool:
        """Ask the LLM whether this query depends on previous context."""
        # Could be replaced by simple heuristic if you don't want LLM cost
        followup_keywords = ["lesser", "cheaper", "bigger", "smaller", "closer", "same area", "more", "fewer"]
        return any(kw in text.lower() for kw in followup_keywords)


    def _merge_queries(self, prev_query: str, followup: str) -> str:
        """Merge the old query with new follow-up constraints."""
        return f"{prev_query}, but {followup}"


    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.last_results = None
        self.last_query_context = None

    def save_conversation(self, filepath: str):
        """Save conversation history to file"""
        history = self.get_conversation_history()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'conversation': history
            }, f, indent=2, ensure_ascii=False)

    def load_conversation(self, filepath: str):
        """Load conversation history from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.clear_memory()
            for msg in data['conversation']:
                if msg['role'] == 'user':
                    self.memory.chat_memory.add_user_message(msg['content'])
                else:
                    self.memory.chat_memory.add_ai_message(msg['content'])
            
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False


# ---------------------------
# Example usage
# ---------------------------
if __name__ == '__main__':
    CHROMA_PATH = r"C:\Users\Pragan Nisha\AppData\Local\Programs\Ollama\chroma_db_rental_detailsS"
    
    # Initialize conversational system
    chat_system = ConversationalPropertySystem(
        chroma_persist_path=CHROMA_PATH,
        gemini_key_search=os.getenv('GEMINI_API_1'),
        gemini_key_analytics=os.getenv('GEMINI_API_2')
    )
    
    print("🏠 Property Assistant: Hello! I can help you search for rental properties and provide market insights.")
    print("💡 Type 'exit' to quit.\n")
    
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("🤖 Assistant: Goodbye! 👋")
            break
        
        response = chat_system.chat(user_input)
        print(f"🤖 Assistant: {response}\n")

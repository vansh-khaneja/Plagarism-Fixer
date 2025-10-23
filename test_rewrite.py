#!/usr/bin/env python3
"""
Test script for the rewrite chunk endpoint
"""

import requests
import json

def test_rewrite_endpoint():
    """Test the rewrite chunk endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/rewrite-chunk"
    
    # Test data - plagiarized chunk
    test_data = {
        "chunk_text": "Artificial intelligence is revolutionizing the way we approach complex problems across various industries. Machine learning algorithms can now process vast amounts of data with unprecedented accuracy and speed.",
        "num_suggestions": 3
    }
    
    print("Testing Rewrite Chunk Endpoint...")
    print("="*60)
    print(f"Original text: {test_data['chunk_text']}")
    print(f"Requesting {test_data['num_suggestions']} suggestions")
    print("-" * 60)
    
    try:
        # Make API request
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS!")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"\n📝 REWRITE SUGGESTIONS:")
            print("-" * 60)
            
            for i, suggestion in enumerate(result['suggestions'], 1):
                print(f"\n🔄 Suggestion {i}:")
                print(f"{suggestion}")
                print(f"Word count: {len(suggestion.split())} words")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            result = response.json()
            print("🏥 HEALTH CHECK:")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Credentials: {result['credentials']}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    print("Testing Rewrite Chunk API...\n")
    test_health()
    print("\n" + "="*60)
    test_rewrite_endpoint()

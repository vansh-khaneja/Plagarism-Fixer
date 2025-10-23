#!/usr/bin/env python3
"""
Test script for the FastAPI plagiarism checker
"""

import requests
import json

def test_api():
    """Test the plagiarism checker API"""
    
    # API endpoint
    url = "http://localhost:8000/check-plagiarism"
    
    # Test data - only content needed now!
    test_data = {
        "content": """
        Artificial intelligence is revolutionizing the way we approach complex problems across various industries. 
        Machine learning algorithms can now process vast amounts of data with unprecedented accuracy and speed. 
        Deep learning networks have enabled breakthroughs in computer vision, natural language processing, and 
        speech recognition. These technologies are being integrated into healthcare systems to assist in medical 
        diagnosis and treatment planning. Financial institutions utilize AI for fraud detection, risk assessment, 
        and algorithmic trading. Educational platforms leverage machine learning to personalize learning 
        experiences and provide adaptive tutoring systems.
        """
    }
    
    print("Testing Plagiarism Checker API...")
    print("="*50)
    
    try:
        # Make API request
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS!")
            print(f"Total words: {result['total_words']}")
            print(f"Total chunks: {result['total_chunks']}")
            print(f"Chunks processed: {result['chunks_processed']}")
            print(f"Chunks with plagiarism: {result['chunks_with_plagiarism']}")
            print(f"Overall plagiarism score: {result['overall_plagiarism_score']:.2f}%")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            
            print(f"\n📋 CHUNK DETAILS:")
            print("-" * 50)
            
            for chunk in result['chunk_details']:
                print(f"\nChunk {chunk['chunk_id']}:")
                print(f"  Words: {chunk['word_count']}")
                print(f"  Plagiarized: {'Yes' if chunk['is_plagiarized'] else 'No'}")
                if chunk['is_plagiarized']:
                    print(f"  Similarity: {chunk['similarity_score']:.1f}%")
                    print(f"  Source: {chunk['source_title']}")
                    print(f"  URL: {chunk['source_url']}")
                print(f"  Content: {chunk['content'][:100]}...")
                
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
            print(f"Configuration: {result['configuration']}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    print("Testing API endpoints...\n")
    test_health()
    print("\n" + "="*50)
    test_api()

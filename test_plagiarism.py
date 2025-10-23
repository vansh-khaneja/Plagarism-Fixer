#!/usr/bin/env python3
"""
Simple test script for the improved plagiarism checker
"""

from plagchecker import PlagiarismChecker

def test_plagiarism_checker():
    """Test the improved plagiarism checker"""
    
    # Your API credentials (replace with actual values)
# Your credentials
    API_KEY = "AIzaSyC-o0bvh4JECQCZsNadG8chL5YhiTpJQsY"
    SEARCH_ENGINE_ID = "d64adfd7ddd454ee5"
    
    print("="*60)
    print("TESTING IMPROVED PLAGIARISM CHECKER")
    print("="*60)
    
    # Initialize checker
    checker = PlagiarismChecker(
        api_key=API_KEY,
        search_engine_id=SEARCH_ENGINE_ID,
        chunk_size=80,
        overlap_percentage=0.15
    )
    
    # Test text
    test_text = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion.

Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and returned to his technology companies.

Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    """
    
    print("Testing with sample text...")
    print(f"Text length: {len(test_text.split())} words")
    
    # Check for plagiarism
    result = checker.check_plagiarism(
        article_text=test_text,
        similarity_threshold=50,
        max_chunks=5,  # Limit for testing
        delay_between_requests=1.0
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Plagiarism percentage: {result['plagiarism_percentage']:.2f}%")
    print(f"Matches found: {result['plagiarized_chunks']}")
    
    if result['matches']:
        print(f"\nDetailed matches:")
        for i, match in enumerate(result['matches'], 1):
            print(f"  {i}. {match['similarity']:.1f}% similar")
            print(f"     Source: {match['source_title']}")
    else:
        print("No plagiarism detected!")

if __name__ == "__main__":
    test_plagiarism_checker()

import requests
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PlagiarismChecker:
    def __init__(self, api_key, search_engine_id, chunk_size=100, overlap_percentage=0.15):
        """
        Initialize plagiarism checker with LangChain-powered chunking
        
        Args:
            api_key: Google Custom Search API key
            search_engine_id: Google Custom Search Engine ID
            chunk_size: Number of words per chunk (default: 100)
            overlap_percentage: Overlap between chunks as percentage (default: 0.15 = 15%)
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_percentage)
        
        # Initialize LangChain text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 5,  # Approximate chars per word
            chunk_overlap=int(chunk_size * 5 * overlap_percentage),
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def create_chunks(self, text):
        """
        Create chunks using LangChain's intelligent text splitting with sentence completion
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with metadata
        """
        # Use LangChain's recursive splitter for better chunking
        chunk_texts = self.text_splitter.split_text(text)
        
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            # Find position in original text
            start_pos = text.find(chunk_text, current_pos)
            end_pos = start_pos + len(chunk_text)
            
            # Ensure sentence completion - extend chunk if it ends mid-sentence
            enhanced_chunk = self._complete_sentence(chunk_text, text, end_pos)
            
            chunks.append({
                'text': enhanced_chunk.strip(),
                'start_word': len(text[:start_pos].split()),
                'end_word': len(text[:start_pos + len(enhanced_chunk)].split()),
                'word_count': len(enhanced_chunk.split()),
                'chunk_id': i
            })
            
            current_pos = start_pos
        
        return chunks
    
    def _complete_sentence(self, chunk_text, full_text, chunk_end_pos):
        """
        Complete the sentence if chunk ends mid-sentence
        
        Args:
            chunk_text: Current chunk text
            full_text: Full original text
            chunk_end_pos: Position where chunk ends in full text
            
        Returns:
            Enhanced chunk text with completed sentence
        """
        # Check if chunk ends with sentence-ending punctuation
        if chunk_text.rstrip().endswith(('.', '!', '?', '."', '!"', '?"')):
            return chunk_text
        
        # Find the next sentence ending after the chunk
        remaining_text = full_text[chunk_end_pos:]
        
        # Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '."', '!"', '?"']
        
        for ending in sentence_endings:
            if ending in remaining_text:
                # Find the position of the sentence ending
                end_pos = remaining_text.find(ending) + len(ending.rstrip())
                # Add the complete sentence to the chunk
                return chunk_text + remaining_text[:end_pos]
        
        # If no sentence ending found, look for paragraph breaks
        if '\n\n' in remaining_text:
            para_end = remaining_text.find('\n\n')
            return chunk_text + remaining_text[:para_end]
        
        # If still no clear ending, extend to next 20 words max to avoid huge chunks
        words = remaining_text.split()
        if len(words) > 0:
            # Add up to 20 more words
            additional_words = min(20, len(words))
            additional_text = ' '.join(words[:additional_words])
            return chunk_text + ' ' + additional_text
        
        return chunk_text
    
    def search_google(self, query, max_retries=3):
        """Search Google with retry logic and better query optimization"""
        url = "https://www.googleapis.com/customsearch/v1"
        
        # Optimize query for better plagiarism detection
        optimized_query = self._optimize_query(query)
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': optimized_query,
            'num': 5
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    if 'items' in data:
                        for item in data['items']:
                            results.append({
                                'title': item.get('title', ''),
                                'link': item.get('link', ''),
                                'snippet': item.get('snippet', '')
                            })
                    return results
                elif response.status_code == 429:  # Rate limit
                    print(f"Rate limited. Waiting {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"API Error: {response.status_code}")
                    return []
                    
            except Exception as e:
                print(f"Search error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return []
    
    def _optimize_query(self, query):
        """Optimize search query for better plagiarism detection"""
        # Remove very common words and use exact phrase matching
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.split()
        
        # Keep important words
        important_words = [word for word in words if word.lower() not in common_words]
        
        # Use exact phrase matching for better results
        if len(important_words) >= 3:
            return f'"{query[:150]}"'  # Limit length
        else:
            return query[:150]
    
    def calculate_similarity(self, text1, text2):
        """Enhanced similarity calculation using multiple methods"""
        try:
            # Method 1: TF-IDF with n-grams
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
                min_df=1,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Method 2: Character-level similarity
            char_similarity = self._char_similarity(text1, text2)
            
            # Method 3: Word overlap
            word_similarity = self._word_overlap(text1, text2)
            
            # Weighted combination for better accuracy
            final_similarity = (
                tfidf_similarity * 0.5 +
                char_similarity * 0.3 +
                word_similarity * 0.2
            )
            
            return min(final_similarity * 100, 100)  # Cap at 100%
            
        except:
            return 0
    
    def _char_similarity(self, text1, text2):
        """Calculate character-level similarity"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _word_overlap(self, text1, text2):
        """Calculate word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def check_plagiarism(self, article_text, similarity_threshold=50, 
                        max_chunks=None, delay_between_requests=1.5):
        """
        Check article for plagiarism using improved chunking strategy
        
        Args:
            article_text: Text to check
            similarity_threshold: Percentage threshold for flagging (default: 50%)
            max_chunks: Maximum chunks to check (None = all, useful for API limits)
            delay_between_requests: Delay in seconds between API calls
            
        Returns:
            Dictionary with plagiarism results
        """
        # Preprocess text
        clean_text = self.preprocess_text(article_text)
        
        # Create chunks using LangChain
        chunks = self.create_chunks(clean_text)
        
        # Limit chunks if specified (for API quota management)
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        total_chunks = len(chunks)
        plagiarized_chunks = []
        checked_chunks = 0
        
        print(f"Created {total_chunks} intelligent chunks using LangChain")
        print(f"Average chunk size: {sum(c['word_count'] for c in chunks) / len(chunks):.1f} words")
        print(f"Starting plagiarism check...\n")
        
        for i, chunk in enumerate(chunks):
            print(f"Checking chunk {i+1}/{total_chunks} ({chunk['word_count']} words)...")
            
            # Search for this chunk
            search_results = self.search_google(chunk['text'])
            
            if search_results:
                max_similarity = 0
                best_match = None
                
                # Compare chunk with each search result
                for result in search_results:
                    snippet = result['snippet']
                    similarity = self.calculate_similarity(chunk['text'], snippet)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = result
                
                # Flag if above threshold
                if max_similarity >= similarity_threshold:
                    plagiarized_chunks.append({
                        'chunk_index': i,
                        'text': chunk['text'][:100] + '...',  # Preview
                        'word_range': f"{chunk['start_word']}-{chunk['end_word']}",
                        'similarity': max_similarity,
                        'source_url': best_match['link'],
                        'source_title': best_match['title']
                    })
                    print(f"  ⚠️  MATCH FOUND: {max_similarity:.1f}% similar")
                else:
                    print(f"  ✅  Original ({max_similarity:.1f}% similarity)")
            else:
                print(f"  ✅  No online matches found")
            
            checked_chunks += 1
            
            # Rate limiting
            if i < total_chunks - 1:  # Don't delay after last chunk
                time.sleep(delay_between_requests)
        
        # Calculate overall plagiarism percentage
        plagiarism_percentage = (len(plagiarized_chunks) / total_chunks) * 100 if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'checked_chunks': checked_chunks,
            'plagiarized_chunks': len(plagiarized_chunks),
            'plagiarism_percentage': plagiarism_percentage,
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'matches': plagiarized_chunks
        }

# Usage example
if __name__ == "__main__":
    # Initialize checker with improved chunking
    checker = PlagiarismChecker(
        api_key="YOUR_API_KEY",
        search_engine_id="YOUR_CX_ID",
        chunk_size=80,      # 80 words per chunk
        overlap_percentage=0.15  # 15% overlap
    )
    
    # Your article
    article = """
    Artificial intelligence is transforming the way we live and work in unprecedented ways.
    Machine learning algorithms can now process vast amounts of data with remarkable accuracy.
    Deep learning has revolutionized computer vision applications across various industries.
    Natural language processing enables computers to understand and generate human language.
    These technologies are being applied in healthcare, finance, education, and many other sectors.
    The future of AI holds tremendous potential for solving complex global challenges.
    """
    
    # Check plagiarism
    result = checker.check_plagiarism(
        article, 
        similarity_threshold=50,
        max_chunks=10,  # Limit for testing
        delay_between_requests=1.2
    )
    
    # Print report
    print("\n" + "="*70)
    print("PLAGIARISM DETECTION REPORT")
    print("="*70)
    print(f"Total chunks analyzed: {result['total_chunks']}")
    print(f"Chunk size: {result['chunk_size']} words")
    print(f"Overlap: {result['overlap_size']} words")
    print(f"Plagiarized chunks: {result['plagiarized_chunks']}")
    print(f"Plagiarism score: {result['plagiarism_percentage']:.2f}%")
    
    if result['matches']:
        print("\n" + "-"*70)
        print("DETAILED MATCHES:")
        print("-"*70)
        for i, match in enumerate(result['matches'], 1):
            print(f"\n{i}. Words {match['word_range']}")
            print(f"   Text: {match['text']}")
            print(f"   Similarity: {match['similarity']:.1f}%")
            print(f"   Source: {match['source_title']}")
            print(f"   URL: {match['source_url']}")

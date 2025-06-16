import logging
import re
import os
from typing import List, Dict, Any, Optional, Union
import uuid
import json

from ..orchestrator import RetryableError

logger = logging.getLogger(__name__)


def run(input_data: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits text into chunks for embedding and indexing.
    
    Args:
        input_data: Either raw text string or a dictionary with text and metadata
        
    Returns:
        List of chunk dictionaries with content and metadata fields
        
    Raises:
        RetryableError: For transient failures that should be retried
    """
    # Parse input
    document_metadata = {}
    if isinstance(input_data, str):
        raw_text = input_data
    elif isinstance(input_data, dict):
        raw_text = input_data.get("text", "")
        # Extract metadata if provided
        document_metadata = {
            "document_type": input_data.get("document_type", "unknown"),
            "document_name": input_data.get("document_name", ""),
            "document_year": input_data.get("document_year", ""),
            "document_entity": input_data.get("document_entity", ""),
            "source_uri": input_data.get("source_uri", ""),
            "document_id": input_data.get("document_id")
        }
    else:
        raise ValueError(f"Invalid input type: {type(input_data)}")
    
    logger.info(f"Chunking text of length {len(raw_text)}")
    
    try:
        # Try to extract document metadata from text if not provided
        if not any(document_metadata.values()):
            extracted_metadata = extract_metadata_from_text(raw_text)
            document_metadata.update(extracted_metadata)
        
        # Extract pages if the text has page markers
        pages = extract_pages(raw_text)
        
        # If no pages were detected, treat the entire text as one page
        if not pages:
            pages = [{"content": raw_text, "page_num": 1}]
        
        # Create chunks from pages
        chunks = []
        for page in pages:
            page_chunks = create_chunks_from_page(
                page["content"], 
                page["page_num"],
                document_metadata,
                max_chunk_size=1000,
                overlap=100
            )
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks
        
    except Exception as e:
        # Handle transient errors
        if is_transient_error(e):
            logger.warning(f"Transient error chunking text: {str(e)}")
            raise RetryableError(f"Transient error: {str(e)}")
        
        # Handle permanent errors
        logger.error(f"Error chunking text: {str(e)}")
        raise


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Attempt to extract document metadata from the text content.
    
    Args:
        text: The document text
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        "document_type": "unknown",
        "document_name": "",
        "document_year": "",
        "document_entity": ""
    }
    
    # Extract document type
    type_patterns = [
        (r'\b(?:policy|policies)\b', 'policy'),
        (r'\b(?:manual|handbook)\b', 'manual'),
        (r'\b(?:report|analysis)\b', 'report'),
        (r'\b(?:contract|agreement)\b', 'contract'),
        (r'\b(?:memo|memorandum)\b', 'memo'),
        (r'\b(?:presentation|slides)\b', 'presentation')
    ]
    
    for pattern, doc_type in type_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            metadata["document_type"] = doc_type
            break
    
    # Extract year (4-digit number that looks like a year)
    year_matches = re.findall(r'\b(19[7-9]\d|20[0-2]\d)\b', text)
    if year_matches:
        metadata["document_year"] = year_matches[0]
    
    # Extract document name (look for title-like text at the beginning)
    first_lines = text.split('\n')[:5]  # Check first 5 lines
    for line in first_lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100 and not line.endswith('.'):
            # Looks like a title
            metadata["document_name"] = line
            break
    
    # Extract entity names (look for company or organization names)
    entity_patterns = [
        r'(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company|GmbH)',
        r'(?:University|College|Institute|Organization|Association)'
    ]
    
    for pattern in entity_patterns:
        entity_regex = fr'\b([A-Z][A-Za-z0-9\'\-\s]+\s+{pattern})\b'
        entity_matches = re.findall(entity_regex, text)
        if entity_matches:
            metadata["document_entity"] = entity_matches[0]
            break
    
    return metadata


def extract_pages(text: str) -> List[Dict[str, Any]]:
    """
    Extract pages from text based on page markers.
    
    Args:
        text: The text to extract pages from
        
    Returns:
        List of dictionaries with "content" and "page_num" fields
    """
    # Try to find page markers in the text
    # Common patterns: "Page X", "- X -", "[Page X]", etc.
    page_patterns = [
        r'\n\s*Page\s+(\d+)\s*\n',
        r'\n\s*-\s*(\d+)\s*-\s*\n',
        r'\n\s*\[\s*Page\s+(\d+)\s*\]\s*\n',
        r'\n\s*<page[^>]*>(\d+)</page>\s*\n',
        r'\f\s*(\d+)\s*'  # Form feed character often used as page separator
    ]
    
    # Try each pattern
    for pattern in page_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Found page markers, split the text
            pages = []
            for i, match in enumerate(matches):
                page_num = int(match.group(1))
                start_pos = 0 if i == 0 else matches[i-1].end()
                end_pos = match.start()
                
                # Skip empty pages
                if start_pos >= end_pos:
                    continue
                
                page_content = text[start_pos:end_pos].strip()
                if page_content:
                    pages.append({
                        "content": page_content,
                        "page_num": page_num
                    })
            
            # Add the last page
            if matches and matches[-1].end() < len(text):
                last_content = text[matches[-1].end():].strip()
                if last_content:
                    last_page_num = int(matches[-1].group(1)) + 1
                    pages.append({
                        "content": last_content,
                        "page_num": last_page_num
                    })
            
            if pages:
                return pages
    
    # If no page markers were found, try to split by form feed character
    if '\f' in text:
        pages = []
        page_parts = text.split('\f')
        for i, page_content in enumerate(page_parts):
            page_content = page_content.strip()
            if page_content:
                pages.append({
                    "content": page_content,
                    "page_num": i + 1
                })
        if pages:
            return pages
    
    # No page markers found
    return []


def create_chunks_from_page(
    page_text: str,
    page_num: int,
    document_metadata: Dict[str, Any],
    max_chunk_size: int = 1000,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Create overlapping chunks from a page of text.
    
    Args:
        page_text: The text of the page
        page_num: The page number
        document_metadata: Metadata about the document
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # If the page is small enough, use it as a single chunk
    if len(page_text) <= max_chunk_size:
        chunk = {
            "content": page_text,
            "page": page_num,
            "chunk_id": str(uuid.uuid4())
        }
        # Add document metadata
        chunk.update(document_metadata)
        chunks.append(chunk)
        return chunks
    
    # Split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n', page_text)
    
    # Create chunks by combining paragraphs
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed the max size, create a new chunk
        if len(current_chunk) + len(paragraph) + 1 > max_chunk_size and current_chunk:
            chunk = {
                "content": current_chunk,
                "page": page_num,
                "chunk_id": str(uuid.uuid4())
            }
            # Add document metadata
            chunk.update(document_metadata)
            chunks.append(chunk)
            
            # Start a new chunk with overlap from the previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                # Try to find a sentence boundary for the overlap
                overlap_text = find_sentence_boundary(current_chunk, overlap)
                current_chunk = overlap_text
            else:
                current_chunk = ""
        
        # Add the paragraph to the current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunk = {
            "content": current_chunk,
            "page": page_num,
            "chunk_id": str(uuid.uuid4())
        }
        # Add document metadata
        chunk.update(document_metadata)
        chunks.append(chunk)
    
    return chunks


def find_sentence_boundary(text: str, target_length: int) -> str:
    """
    Find a sentence boundary near the target length from the end of the text.
    
    Args:
        text: The text to search in
        target_length: The target length from the end
        
    Returns:
        Text from the sentence boundary to the end
    """
    if len(text) <= target_length:
        return text
    
    # Get the last part of the text
    end_text = text[-target_length:]
    
    # Look for sentence boundaries (., !, ?)
    sentence_boundaries = [m.start() for m in re.finditer(r'[.!?]\s+', end_text)]
    
    if sentence_boundaries:
        # Find the first sentence boundary
        first_boundary = sentence_boundaries[0]
        return end_text[first_boundary+2:]  # +2 to skip the punctuation and space
    
    # If no sentence boundary found, look for a space
    space_positions = [m.start() for m in re.finditer(r'\s+', end_text)]
    
    if space_positions:
        # Find the first space
        first_space = space_positions[0]
        return end_text[first_space+1:]  # +1 to skip the space
    
    # If no space found, just return the end text
    return end_text


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.
    """
    # Check for common transient error patterns
    error_str = str(error).lower()
    
    # Memory or processing errors that might be transient
    if any(msg in error_str for msg in ['memory', 'timeout', 'temporarily']):
        return True
    
    return False

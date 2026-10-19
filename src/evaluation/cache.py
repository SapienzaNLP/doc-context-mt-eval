"""Prompt caching utilities for response storage and retrieval."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PromptCache:
    """Cache for storing and retrieving model responses."""
    
    def __init__(self, cache_dir: str = "prompt_cache", enabled: bool = True):
        """
        Initialize prompt cache.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Prompt cache initialized at: {self.cache_dir}")
        else:
            logger.info("Prompt cache disabled")
    
    def get_cache_filename(self, prompt: str) -> Path:
        """
        Generate a unique filename for a prompt using MD5 hash.
        """
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{prompt_hash}.json"


    def load(self, prompt: str) -> Optional[Tuple[str, str]]:
        """
        Load a model response from cache if it exists.
        """
        if not self.enabled:
            return None
        
        cache_file = self.get_cache_filename(prompt)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            thinking_content = data.get("thinking_content", "")
            response = data.get("response", "")
            
            logger.debug(f"Cache hit for prompt hash: {cache_file.stem}")
            return thinking_content, response
            
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            return None


    def save(self, prompt: str, response: str, thinking_content: str = "") -> None:
        """
        Save a model response to cache.
        
        Args:
            prompt: The prompt string
            response: The model's response
            thinking_content: Any additional "thinking" content from the model
        """
        if not self.enabled:
            return
        
        cache_file = self.get_cache_filename(prompt)
        
        data = {
            "prompt": prompt,
            "response": response,
            "thinking_content": thinking_content
        }
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached response for prompt hash: {cache_file.stem}")
            
        except IOError as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")


    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled or not self.cache_dir.exists():
            return {
                "enabled": self.enabled,
                "total_entries": 0,
                "total_size_mb": 0
            }
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": self.enabled,
            "total_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
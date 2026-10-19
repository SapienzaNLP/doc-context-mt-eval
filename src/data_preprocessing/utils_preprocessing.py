"""Utility functions for WMT preprocessing."""

from typing import List, Dict, Any, Optional


class RatingSerializer:
    """Handles serialization of rating objects."""
    
    @staticmethod
    def serialize(rating: Any) -> Optional[List[Dict]]:
        """
        Serialize a rating object into a list of error dictionaries.
        Returns None if rating is None, or an empty list if no errors exist.
        """
        if rating is None:
            return None
        
        errors = getattr(rating, "errors", [])
        if not errors:
            return []
        
        return [
            {
                "start": err.start,
                "end": err.end,
                "category": err.category,
                "severity": err.severity,
                "score": err.score,
                "is_source_error": err.is_source_error
            }
            for err in errors
        ]
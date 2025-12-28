"""
Juris-RAG 源代码包
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_processing import LegalDataProcessor
from .rag_engine import JurisRAGEngine, get_rag_engine

__all__ = ["LegalDataProcessor", "JurisRAGEngine", "get_rag_engine"]

"""
Centralized application configuration.

Enterprise systems avoid hardcoded values.
Configurations should be managed centrally.
"""

# =========================================================
# PDF Configuration
# =========================================================

PDF_PATH = "data/documents/wiser-provider-supplier-guide.pdf"

# =========================================================
# Chunking Configuration
# =========================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

"""
Embedding Configuration
"""

EMBEDDING_MODEL = "text-embedding-3-small"

"""
Embedding Batch Configuration
"""

EMBEDDING_BATCH_SIZE = 100
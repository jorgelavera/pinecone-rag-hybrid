"""Exception taxonomy for the retrieval pipeline.

One root (`RagError`) keeps CLI handlers simple: only configuration problems are
translated into exit code 2, everything else surfaces as a plain traceback so a
real defect is never disguised as a setup mistake.

Standard library only — this module is part of the credential-free tier.
"""


class RagError(Exception):
    """Root for every failure mode this project names explicitly."""


class ConfigError(RagError):
    """Environment/configuration is present but invalid."""


class MissingConfigError(ConfigError):
    """One or more required environment variables are absent."""


class CorpusTooSmallError(RagError):
    """Fewer corpus documents loaded than the configured floor."""


class MetadataTooLargeError(RagError):
    """Chunk metadata exceeds the Pinecone 40,000-byte record limit."""


class DimensionMismatchError(RagError):
    """An embedding vector does not match the index dimension (1536)."""


class IndexConfigMismatchError(RagError):
    """The existing index has a different dimension or metric than expected."""


class GoldenSetError(RagError):
    """The Golden Set is missing, malformed, or semantically invalid."""


class HydrationError(RagError):
    """The BM25 corpus could not be rebuilt from Pinecone or the local corpus."""


class RetrievalEmptyError(RagError):
    """A query returned zero documents; surfaced instead of an empty list."""

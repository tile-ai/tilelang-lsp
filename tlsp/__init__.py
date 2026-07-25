"""tlsp - TileLang Language Server.

A language server for TileLang providing inlay hints (buffer shape/dtype/scope,
inferred layouts), hover details and precise compiler diagnostics.

Architecture:
    LSP client <-- LSP/stdio --> tlsp.server (pygls)
                                   <-- NDJSON/stdio --> tlsp.analyzer (persistent
                                   worker subprocess that imports tilelang)
"""

__version__ = "0.2.0"

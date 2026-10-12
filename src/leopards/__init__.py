"""Shared helpers for the Chitabe leopard analysis pipeline.

The numbered scripts in ``src/`` run from the repository root, so ``src/`` is on
``sys.path`` and these modules import as ``from leopards.<module> import ...``.
Everything that used to be copy-pasted across scripts (the Folium palette, the
primary-leopard/sex helpers, the meters projection + MCP geometry, the season
mapping, config constants, and zip/exiftool resolution) lives here once.
"""

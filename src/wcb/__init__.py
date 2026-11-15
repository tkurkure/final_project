"""Cross-bank monetary policy stance detection (CS 72 final project).

Measures how much a stance classifier degrades on a central bank it was never trained on
(the *transfer gap*), decomposes that gap into "less data" vs "this specific bank is
missing", and asks which features of a bank's language predict it.
"""
__version__ = "0.1.0"

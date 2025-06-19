"""
Utility functions for the optical network simulation application.
"""
def dbm2mw(dbm):
    """Convierte dBm a mW."""
    return 10 ** (dbm / 10) 
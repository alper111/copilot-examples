"""A VSCode plugin for Spotify. It allows you to control Spotify from VSCode."""

import os
import sys
import json
import time
import socket
import argparse
import subprocess
import webbrowser
from urllib.parse import quote
from urllib.request import urlopen

__version__ = "0.1.0"

# Spotify URI
SPOTIFY_URI = "spotify:"
# Spotify URI prefix
SPOTIFY_URI_PREFIX = "spotify:"
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)
# Spotify URI prefix length
SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)

# Spotify URI prefix length

SPOTIFY_URI_PREFIX_LENGTH = len(SPOTIFY_URI_PREFIX)


def get_spotify_uri(uri):
    """Get Spotify URI."""
    if uri.startswith(SPOTIFY_URI_PREFIX):
        return uri
    return SPOTIFY_URI + uri


def get_spotify_uri_prefix_length():
    """Get Spotify URI prefix length."""
    return SPOTIFY_URI_PREFIX_LENGTH


def get_spotify_uri_prefix():
    """Get Spotify URI prefix."""
    return SPOTIFY_URI_PREFIX


def get_spotify_uri_prefix():
    """Get Spotify URI prefix."""
    return SPOTIFY_URI_PREFIX


def get_spotify_uri_prefix():
    """Get Spotify URI prefix."""
    return SPOTIFY_URI_PREFIX


def get_spotify_uri_prefix():
    """Get Spotify URI prefix."""
    return SPOTIFY_URI_PREFIX

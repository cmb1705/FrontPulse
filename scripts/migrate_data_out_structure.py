#!/usr/bin/env python3
"""
Migrate data/out/ directory structure to organized pipeline stages.

CRITICAL: This script will:
1. Analyze all scripts for data/out/ references
2. Create path migration mapping
3. Update all scripts with new paths
4. Move existing files to new locations
5. Validate pipeline integrity

Run with --dry-run first to preview changes!

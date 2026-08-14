#!/usr/bin/env python
"""Fix Unicode characters in run_analytics.py for Windows console compatibility."""

with open('analytics/run_analytics.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace problematic Unicode with ASCII equivalents
replacements = {
    '✓': '[OK]',
    '✗': '[X]',
    '•': '*',
    '≈': '~',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('analytics/run_analytics.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed Unicode characters in run_analytics.py')

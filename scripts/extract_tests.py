#!/usr/bin/env python3
"""
Extract OCCT GTests and convert to Rust test specifications.
This creates the ground truth for our port.
"""

import re
import os
import sys
from pathlib import Path

OCCT_SRC = Path("/home/heim/projects/occt-source/src")

def extract_tests_from_cxx(filepath: Path) -> list[dict]:
    """Extract TEST() blocks from a GTest file."""
    tests = []
    content = filepath.read_text()
    
    # Match TEST(SuiteName, TestName) { ... }
    pattern = r'TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        suite, name, body = match.groups()
        tests.append({
            'suite': suite,
            'name': name,
            'body': body.strip(),
            'file': str(filepath)
        })
    
    return tests

def cpp_to_rust_assertion(line: str) -> str:
    """Convert C++ assertion to Rust equivalent."""
    line = line.strip()
    
    # EXPECT_TRUE(x) -> assert!(x)
    if m := re.match(r'EXPECT_TRUE\s*\((.+)\)', line):
        return f'assert!({m.group(1)});'
    
    # EXPECT_FALSE(x) -> assert!(!x)
    if m := re.match(r'EXPECT_FALSE\s*\((.+)\)', line):
        return f'assert!(!{m.group(1)});'
    
    # EXPECT_DOUBLE_EQ(a, b) -> assert!((a - b).abs() < f64::EPSILON)
    if m := re.match(r'EXPECT_DOUBLE_EQ\s*\((.+),\s*(.+)\)', line):
        return f'assert!(({m.group(1)} - {m.group(2)}).abs() < f64::EPSILON);'
    
    # EXPECT_NEAR(a, b, tol) -> assert!((a - b).abs() < tol)
    if m := re.match(r'EXPECT_NEAR\s*\((.+),\s*(.+),\s*(.+)\)', line):
        return f'assert!(({m.group(1)} - {m.group(2)}).abs() < {m.group(3)});'
    
    # EXPECT_EQ(a, b) -> assert_eq!(a, b)
    if m := re.match(r'EXPECT_EQ\s*\((.+),\s*(.+)\)', line):
        return f'assert_eq!({m.group(1)}, {m.group(2)});'
    
    return f'// TODO: {line}'

def generate_rust_test(test: dict) -> str:
    """Generate Rust test from extracted C++ test."""
    rust_name = f"test_{test['suite'].lower()}_{test['name'].lower()}"
    
    # Convert body line by line
    rust_body_lines = []
    for line in test['body'].split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        if 'EXPECT_' in line or 'ASSERT_' in line:
            rust_body_lines.append(f'    {cpp_to_rust_assertion(line)}')
        else:
            # Variable declarations, method calls, etc need manual translation
            rust_body_lines.append(f'    // C++: {line}')
    
    rust_body = '\n'.join(rust_body_lines)
    
    return f'''#[test]
fn {rust_name}() {{
    // Source: {test['file']}
{rust_body}
}}
'''

def find_gtest_files(package: str) -> list[Path]:
    """Find all GTest files for a package."""
    results = []
    for gtest_dir in OCCT_SRC.rglob("GTests"):
        for f in gtest_dir.glob(f"{package}*_Test.cxx"):
            results.append(f)
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: extract_tests.py <package>")
        print("Example: extract_tests.py Bnd")
        sys.exit(1)
    
    package = sys.argv[1]
    files = find_gtest_files(package)
    
    if not files:
        print(f"No GTest files found for package: {package}")
        sys.exit(1)
    
    all_tests = []
    for f in files:
        print(f"Processing: {f}")
        tests = extract_tests_from_cxx(f)
        all_tests.extend(tests)
    
    print(f"\nFound {len(all_tests)} tests")
    print("\n// Generated Rust tests:\n")
    
    for test in all_tests:
        print(generate_rust_test(test))

if __name__ == "__main__":
    main()

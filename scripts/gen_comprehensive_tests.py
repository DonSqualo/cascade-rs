#!/usr/bin/env python3
"""
Generate comprehensive tests from OCCT source code.

Reads every method in a .hxx file and generates test cases.
Does NOT rely on their GTests - reads the actual implementation.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

OCCT_SRC = Path("/home/heim/projects/occt-source/src")

@dataclass
class Method:
    name: str
    return_type: str
    params: str
    is_const: bool
    is_static: bool
    body: Optional[str]  # inline body if available
    doc: str

def extract_methods(hxx_content: str, class_name: str) -> List[Method]:
    """Extract all methods from a header file."""
    methods = []
    
    # Pattern for method declarations
    # Handles: constexpr, static, inline, [[nodiscard]], return type, name, params, const, noexcept
    method_pattern = re.compile(
        r'(?:\/\/[^\n]*\n)*'  # Doc comments
        r'(?:Standard_EXPORT\s+)?'
        r'(?:\[\[nodiscard\]\]\s+)?'
        r'(?:static\s+)?'
        r'(?:constexpr\s+)?'
        r'(?:inline\s+)?'
        r'([\w:*&<>\s]+?)\s+'  # Return type
        r'(\w+)\s*'  # Method name
        r'\(([^)]*)\)\s*'  # Parameters
        r'(const)?\s*'  # const qualifier
        r'(?:noexcept)?\s*'  # noexcept
        r'(?:;|{[^}]*}|\{)',  # End (semicolon or inline body)
        re.MULTILINE | re.DOTALL
    )
    
    for match in method_pattern.finditer(hxx_content):
        ret_type = match.group(1).strip()
        name = match.group(2)
        params = match.group(3).strip()
        is_const = bool(match.group(4))
        
        # Skip constructors, destructors, operators for now
        if name.startswith('~') or name == class_name:
            continue
        if name.startswith('operator'):
            continue
            
        methods.append(Method(
            name=name,
            return_type=ret_type,
            params=params,
            is_const=is_const,
            is_static='static' in match.group(0)[:50],
            body=None,
            doc=""
        ))
    
    return methods

def generate_rust_test(class_name: str, method: Method) -> str:
    """Generate a Rust test for a method."""
    test_name = f"test_{class_name.lower()}_{method.name.lower()}"
    
    # Generate test based on method signature
    test_body = []
    test_body.append(f"    // TODO: Port from OCCT {class_name}::{method.name}")
    test_body.append(f"    // Params: {method.params}")
    test_body.append(f"    // Returns: {method.return_type}")
    test_body.append(f"    // Const: {method.is_const}")
    test_body.append("    unimplemented!()")
    
    return f'''#[test]
#[ignore = "not yet implemented"]
fn {test_name}() {{
{chr(10).join(test_body)}
}}
'''

def analyze_class(hxx_path: Path) -> dict:
    """Analyze a header file and return method coverage info."""
    content = hxx_path.read_text()
    
    # Extract class name
    class_match = re.search(r'class\s+(\w+)', content)
    if not class_match:
        return {}
    
    class_name = class_match.group(1)
    methods = extract_methods(content, class_name)
    
    return {
        'class_name': class_name,
        'file': str(hxx_path),
        'methods': methods,
        'method_count': len(methods)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: gen_comprehensive_tests.py <header.hxx>")
        print("Example: gen_comprehensive_tests.py gp_XYZ.hxx")
        sys.exit(1)
    
    hxx_name = sys.argv[1]
    
    # Find the file
    matches = list(OCCT_SRC.rglob(hxx_name))
    if not matches:
        print(f"File not found: {hxx_name}")
        sys.exit(1)
    
    hxx_path = matches[0]
    print(f"Analyzing: {hxx_path}")
    
    info = analyze_class(hxx_path)
    if not info:
        print("Could not parse class")
        sys.exit(1)
    
    print(f"\nClass: {info['class_name']}")
    print(f"Methods found: {info['method_count']}")
    print("\nMethods:")
    for m in info['methods']:
        const_str = " const" if m.is_const else ""
        static_str = "static " if m.is_static else ""
        print(f"  {static_str}{m.return_type} {m.name}({m.params}){const_str}")
    
    print(f"\n// Generated tests for {info['class_name']}:\n")
    for m in info['methods']:
        print(generate_rust_test(info['class_name'], m))

if __name__ == "__main__":
    main()

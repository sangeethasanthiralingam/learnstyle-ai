#!/usr/bin/env python3
"""
Test all pages for accessibility
"""

import requests

def test_pages():
    """Test all pages for accessibility"""
    print("🌐 Testing all pages...")
    
    base_url = "http://localhost:5000"
    pages = [
        ("/", "Home page"),
        ("/login", "Login page"),
        ("/register", "Registration page"),
        ("/quiz", "Quiz page"),
        ("/docs", "Documentation"),
        ("/test-permissions", "Permission test page")
    ]
    
    for page, description in pages:
        try:
            response = requests.get(f"{base_url}{page}")
            if response.status_code == 200:
                print(f"    ✅ {description}: {page}")
            else:
                print(f"    ⚠️  {description}: {page} (status: {response.status_code})")
        except Exception as e:
            print(f"    ❌ {description}: {page} (error: {e})")

if __name__ == "__main__":
    test_pages()

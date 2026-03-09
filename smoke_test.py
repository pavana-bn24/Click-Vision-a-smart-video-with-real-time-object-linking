"""
Simple smoke test to verify that required packages import successfully.
Run inside your virtualenv: python smoke_test.py
"""

modules = [
    ("flask", "Flask"),
    ("PIL", "Image"),
    ("numpy", "array"),
    ("cv2", "VideoCapture"),
    ("pygame", "init"),
    ("torch", "tensor"),
    ("torchvision", "models"),
]

failed = []
for mod_name, symbol in modules:
    try:
        mod = __import__(mod_name)
        print(f"OK: imported {mod_name}")
    except Exception as e:
        print(f"FAIL: could not import {mod_name}: {e}")
        failed.append((mod_name, str(e)))

if failed:
    print("\nSmoke test failed. Install missing packages listed above and re-run.")
else:
    print("\nSmoke test passed. All imports succeeded.")

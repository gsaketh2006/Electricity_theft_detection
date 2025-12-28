#!/usr/bin/env python3
"""Poll /ready until it returns HTTP 200 or timeout is reached.
Usage: python tools/poll_ready.py [--url http://127.0.0.1:5000/ready] [--timeout 60]
"""
import argparse
import sys
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:5000/ready')
parser.add_argument('--timeout', type=int, default=60)
args = parser.parse_args()

end = time.time() + args.timeout
while time.time() < end:
    try:
        with urllib.request.urlopen(args.url, timeout=5) as r:
            code = r.getcode()
            body = r.read().decode('utf-8')
            print(f"HTTP {code}: {body}")
            if code == 200:
                print('READY')
                sys.exit(0)
    except Exception as e:
        print(f"Not ready yet: {e}")
    time.sleep(5)
print('Timeout waiting for ready')
sys.exit(1)

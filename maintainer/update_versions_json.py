import json
import os

URL = "lilyminium.github.io/mdanalysis/"

VERSION = os.environ['VERSION']

with open('versions.json', 'r') as f:
    versions = json.loads(f)

existing = [item['version'] for item in versions]
already_exists = VERSION in existing

if not already_exists:
    for ver in versions:
        ver['latest'] = False

    versions.append({
        'version': VERSION,
        'display': VERSION,
        'url': os.path.join(URL, VERSION),
        'latest': True
        })

with open("versions.json", 'w') as f:
    json.dump(versions, f, indent=2)


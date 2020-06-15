import json
import os
import xml.etree.ElementTree as ET

try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen

<< << << < HEAD
# ========= WRITE JSON =========
URL = os.environ['URL']

== == == =
URL = os.environ['URL']
>>>>>> > versioned-docs
VERSION = os.environ['VERSION']


def get_web_file(filename, callback):
    url = os.path.join(URL, filename)
    try:
        page = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urlopen(page).read().decode()
    except Exception as e:
        print(e)
        try:
            with open(filename, 'r') as f:
                return callback(f)
        except IOError as e:
            print(e)
            versions = []
    else:
        return callback(data)


# ========= WRITE JSON =========
versions = get_web_file('versions.json', json.loads)
existing = [item['version'] for item in versions]
already_exists = VERSION in existing

if not already_exists:
    latest = 'dev' not in VERSION
    if latest:
        for ver in versions:
            ver['latest'] = False

    versions.append({
        'version': VERSION,
        'display': VERSION,
        'url': os.path.join(URL, VERSION),
        'latest': latest
    })

with open("versions.json", 'w') as f:
    json.dump(versions, f, indent=2)

# ========= WRITE HTML STUBS =========
REDIRECT = """
<!DOCTYPE html>
<meta charset="utf-8">
<title>Redirecting to {url}</title>
<meta http-equiv="refresh" content="0; URL={url}">
<link rel="canonical" href="{url}">
"""

for ver in versions[::-1]:
    if ver['latest']:
        latest_url = ver['url']
        break
else:
    try:
        latest_url = versions[-1]['url']
    except IndexError:
        latest_url = URL

for ver in versions[::-1]:
    if 'dev' in ver['version']:
        dev_url = ver['url']
        break
else:
    try:
        dev_url = versions[-1]['url']
    except IndexError:
        dev_url = URL

with open('index.html', 'w') as f:
    f.write(REDIRECT.format(url=latest_url))

with open('latest/index.html', 'w') as f:
    f.write(REDIRECT.format(url=latest_url))

with open('dev/index.html', 'w') as f:
    f.write(REDIRECT.format(url=dev_url))

# ========= WRITE SUPER SITEMAP.XML =========
ET.register_namespace('xhtml', "http://www.w3.org/1999/xhtml")
bigroot = ET.Element("urlset")

# so we could make 1 big sitemap as commented
# below, but they must be max 50 MB / 50k URL.
# Yes, this is 100+ releases, but who knows when
# that'll happen and who'll look at this then?
# bigroot.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")
# for ver in versions:
#     tree = get_web_file(ver['version']+'/sitemap.xml', ET.fromstring)
#     root = tree.getroot()
#     bigroot.extend(root.getchildren())

# so instead we make a sitemap of sitemaps.
bigroot.set("sitemapindex", "http://www.sitemaps.org/schemas/sitemap/0.9")
for ver in versions:
    path = os.path.join(URL, '{}/sitemap.xml'.format(ver['version']))
    sitemap = ET.SubElement(bigroot, 'sitemap')
    ET.SubElement(sitemap, 'loc').text = path

ET.ElementTree(bigroot).write('sitemap.xml',
                              xml_declaration=True,
                              encoding='utf-8',
                              method="xml")

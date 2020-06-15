import os
import glob
import xml.etree.ElementTree as ET

filename = 'sitemap.xml'
ET.register_namespace('xhtml', "http://www.w3.org/1999/xhtml")
bigroot = ET.Element("urlset")
sitemaps = glob.glob('*/sitemap.xml')

# so we could make 1 big sitemap as commented
# below, but they must be max 50 MB / 50k URL.
# Yes, this is 100+ releases, but who knows when
# that'll happen and who'll look at this then?
# bigroot.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")
# for smap in sitemaps:
#     subver = smap.split('/')[-2]
#     tree = ET.parse(smap)
#     root = tree.getroot()
#     bigroot.extend(root.getchildren())

# so instead we make a sitemap of sitemaps.
URL = os.environ['URL']
bigroot.set("sitemapindex", "http://www.sitemaps.org/schemas/sitemap/0.9")
for smap in sitemaps:
    subpath = '/'.join(smap.split('/')[-2:])
    path = os.path.join(URL, subpath)
    sitemap = ET.SubElement(bigroot, 'sitemap')
    ET.SubElement(sitemap, 'loc').text = path

ET.ElementTree(bigroot).write(filename,
                              xml_declaration=True,
                              encoding='utf-8',
                              method="xml")

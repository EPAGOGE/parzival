import sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

ids = sys.argv[1:]
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
    {"id_list": ",".join(ids), "max_results": str(len(ids))})
raw = urllib.request.urlopen(url, timeout=60).read()
root = ET.fromstring(raw)
found = set()
for e in root.findall('a:entry', NS):
    eid = e.find('a:id', NS).text
    title = " ".join(e.find('a:title', NS).text.split())
    pub = e.find('a:published', NS).text[:10]
    upd = e.find('a:updated', NS).text[:10]
    auth = [a.find('a:name', NS).text for a in e.findall('a:author', NS)]
    jr = e.find('arxiv:journal_ref', NS)
    doi = e.find('arxiv:doi', NS)
    com = e.find('arxiv:comment', NS)
    short = eid.split('/abs/')[-1]
    found.add(short.split('v')[0])
    print(f"[{short}] {pub} (upd {upd})")
    print(f"   TITLE: {title}")
    print(f"   AUTH : {'; '.join(auth)}")
    if jr is not None: print(f"   JREF : {' '.join(jr.text.split())}")
    if doi is not None: print(f"   DOI  : {doi.text}")
    if com is not None: print(f"   COMM : {' '.join(com.text.split())[:300]}")
    print()
missing = [i for i in ids if i not in found and not any(i in f for f in found)]
if missing:
    print("!!! NOT RETURNED:", missing)

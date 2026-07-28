import urllib.request, xml.etree.ElementTree as ET, time
ids = ["2102.06663","2410.21765","2505.20988","2605.29866","2308.12197","2509.14185",
       "2210.07191","2305.05660","2405.10916","2510.10090","2107.05870","2201.06780",
       "2603.25104","2604.01868","2401.14615"]
NS={'a':'http://www.w3.org/2005/Atom','ar':'http://arxiv.org/schemas/atom'}
url="http://export.arxiv.org/api/query?id_list="+",".join(ids)+"&max_results=40"
d=urllib.request.urlopen(url,timeout=60).read()
r=ET.fromstring(d)
for e in r.findall('a:entry',NS):
    aid=e.find('a:id',NS).text.split('/abs/')[-1]
    ti=" ".join(e.find('a:title',NS).text.split())
    au=[a.find('a:name',NS).text for a in e.findall('a:author',NS)]
    pub=e.find('a:published',NS).text[:10]; upd=e.find('a:updated',NS).text[:10]
    jr=e.find('ar:journal_ref',NS); doi=e.find('ar:doi',NS)
    print("="*100)
    print(f"ID:      {aid}")
    print(f"TITLE:   {ti}")
    print(f"AUTHORS: {'; '.join(au) if len(au)<25 else '; '.join(au[:6])+f' ... [{len(au)} total]'}")
    print(f"DATES:   published {pub} | updated {upd}")
    print(f"JOURNAL: {jr.text if jr is not None else '(none listed)'}")
    print(f"DOI:     {doi.text if doi is not None else '(none listed)'}")

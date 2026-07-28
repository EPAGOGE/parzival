import urllib.request, json, urllib.parse
qs = [
 ("MMS Hou-Huang degenerate viscosity","Potential singularity formation incompressible axisymmetric Euler degenerate viscosity coefficients"),
 ("Physica D Hou-Huang two-scale traveling wave","potential two-scale traveling wave singularity for 3D incompressible Euler equations"),
 ("PRL Wang-Lai-GomezSerrano-Buckmaster","Asymptotic self-similar blow-up profile three-dimensional axisymmetric Euler equations neural networks"),
]
for label,q in qs:
    u="https://api.crossref.org/works?rows=3&query.bibliographic="+urllib.parse.quote(q)
    req=urllib.request.Request(u,headers={"User-Agent":"citation-verify/1.0 (mailto:hill.jt@icloud.com)"})
    items=json.load(urllib.request.urlopen(req,timeout=45))["message"]["items"]
    print("="*95); print("QUERY:",label)
    for m in items:
        au="; ".join(a.get("family","?") for a in m.get("author",[])[:5])
        yr=m.get("published",{}).get("date-parts",[[None]])[0][0]
        print(f"  TITLE:  {m.get('title',['?'])[0][:95]}")
        print(f"  AUTH:   {au}")
        print(f"  VENUE:  {(m.get('container-title') or ['?'])[0]}  vol {m.get('volume','?')} iss {m.get('issue','?')} pp {m.get('page','?')} art {m.get('article-number','?')} yr {yr}")
        print(f"  DOI:    {m.get('DOI')}\n")

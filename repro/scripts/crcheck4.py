import json, time, urllib.request, urllib.parse
MAIL="hill.jt@icloud.com"
def q(bib, rows=3):
    url="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":bib,"rows":str(rows),"mailto":MAIL})
    req=urllib.request.Request(url, headers={"User-Agent":f"ref-verify/1.0 (mailto:{MAIL})"})
    d=json.load(urllib.request.urlopen(req,timeout=60))
    print("### Q:",bib)
    for it in d["message"]["items"]:
        t=(it.get("title") or ["?"])[0]
        a="; ".join(f"{x.get('given','')} {x.get('family','')}".strip() for x in it.get("author",[]))[:170]
        c=(it.get("container-title") or ["?"])[0]
        yr=None
        for k in ("published-print","published","issued"):
            if it.get(k,{}).get("date-parts",[[None]])[0][0]: yr=it[k]["date-parts"][0][0]; break
        print(f"  - {t[:125]}\n    {a}\n    {c} vol {it.get('volume')} p {it.get('page')} ({yr}) doi:{it.get('DOI')}")
    print()
for s in [
 "Choi Kiselev Yao Finite time blow up for a 1D model of 2D Boussinesq system Communications in Mathematical Physics 2015",
 "Choi Hou Kiselev Sverak Tao Yao On the finite-time blowup of a 1D model for the 3D axisymmetric Euler equations",
 "Hou Liu Self-similar singularity of a 1D model for the 3D axisymmetric Euler equations Research in the Mathematical Sciences 2015",
 "Rica Potential anisotropic finite-time singularity in the three-dimensional axisymmetric Euler equations Physical Review Fluids 2022",
 "Cordoba Martinez-Zoroa Ozanski Instantaneous continuous loss of regularity for the SQG equation",
 "Elgindi Pasqualotto From instability to singularity formation in incompressible fluids",
]:
    try: q(s)
    except Exception as e: print("### Q:",s,"\n ERROR",e,"\n")
    time.sleep(0.6)

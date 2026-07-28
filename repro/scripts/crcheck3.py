import json, time, urllib.request, urllib.parse
MAIL="hill.jt@icloud.com"
def q(bib, rows=3):
    url="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":bib,"rows":str(rows),"mailto":MAIL})
    req=urllib.request.Request(url, headers={"User-Agent":f"ref-verify/1.0 (mailto:{MAIL})"})
    d=json.load(urllib.request.urlopen(req,timeout=60))
    print("### QUERY:",bib)
    for it in d["message"]["items"]:
        t=(it.get("title") or ["?"])[0]
        a="; ".join(f"{x.get('given','')} {x.get('family','')}".strip() for x in it.get("author",[]))[:170]
        c=(it.get("container-title") or ["?"])[0]
        yr=None
        for k in ("published-print","published","issued"):
            if it.get(k,{}).get("date-parts",[[None]])[0][0]: yr=it[k]["date-parts"][0][0]; break
        print(f"  - {t[:130]}\n    {a}\n    {c} vol {it.get('volume')} p {it.get('page')} ({yr}) doi:{it.get('DOI')}")
    print()
for s in [
 "Guillod Sverak Numerical investigations of non-uniqueness for the Navier-Stokes initial value problem in borderline spaces",
 "Hou Li Blowup or no blowup? The interplay between theory and numerics Physica D 2008",
 "Brachet Meiron Orszag Nickel Morf Frisch Small-scale structure of the Taylor-Green vortex Journal of Fluid Mechanics 1983",
 "Papanicolaou Sulem Sulem Wang Dynamic rescaling for tracking point singularities application to nonlinear Schrodinger equation",
 "Landman Papanicolaou Sulem Sulem Wang Stability of isotropic singularities for the nonlinear Schrodinger equation Physica D 1991",
 "Chen Hou Stable nearly self-similar blowup 2D Boussinesq 3D Euler smooth data Part I Annals of PDE",
]:
    try: q(s)
    except Exception as e: print("### QUERY:",s,"\n ERROR",e,"\n")
    time.sleep(0.6)

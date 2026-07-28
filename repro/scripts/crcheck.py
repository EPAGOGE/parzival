import json, sys, time, urllib.request, urllib.parse

MAIL = "hill.jt@icloud.com"

def q(bib, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": bib, "rows": str(rows), "mailto": MAIL})
    req = urllib.request.Request(url, headers={"User-Agent": f"ref-verify/1.0 (mailto:{MAIL})"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    print("### QUERY:", bib)
    for it in d["message"]["items"]:
        title = (it.get("title") or ["?"])[0]
        auth = "; ".join(f"{a.get('given','')} {a.get('family','')}".strip()
                         for a in it.get("author", []))[:180]
        cont = (it.get("container-title") or ["?"])[0]
        yr = None
        for k in ("published-print", "published", "issued"):
            if it.get(k, {}).get("date-parts", [[None]])[0][0]:
                yr = it[k]["date-parts"][0][0]; break
        print(f"  - {title[:130]}")
        print(f"    {auth}")
        print(f"    {cont} vol {it.get('volume')} iss {it.get('issue')} p {it.get('page')} ({yr}) doi:{it.get('DOI')}")
    print()

QUERIES = [
 "Leray Sur le mouvement d'un liquide visqueux emplissant l'espace Acta Mathematica 1934",
 "Beale Kato Majda Remarks on the breakdown of smooth solutions for the 3-D Euler equations",
 "Constantin Lax Majda A simple one-dimensional model for the three-dimensional vorticity equation",
 "De Gregorio On a one-dimensional model for the three-dimensional vorticity equation Journal of Statistical Physics 1990",
 "Sulem Sulem Frisch Tracing complex singularities with spectral methods Journal of Computational Physics 1983",
 "McLaughlin Papanicolaou Sulem Sulem Focusing singularity of the cubic Schrodinger equation Physical Review A 1986",
 "Landman Papanicolaou Sulem Sulem Rate of blowup for solutions of the nonlinear Schrodinger equation at critical dimension",
 "Kerr Evidence for a singularity of the three-dimensional incompressible Euler equations Physics of Fluids 1993",
 "Pumir Siggia Development of singular solutions to the axisymmetric Euler equations Physics of Fluids A 1992",
 "Grauer Sideris Numerical computation of 3D incompressible ideal fluids with swirl Physical Review Letters 1991",
]
for s in QUERIES:
    try:
        q(s)
    except Exception as e:
        print("### QUERY:", s, "\n  ERROR", e, "\n")
    time.sleep(0.6)

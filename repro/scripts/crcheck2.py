import json, time, urllib.request, urllib.parse
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
 "Weinan E Chi-Wang Shu Small-scale structures in Boussinesq convection Physics of Fluids 1994",
 "Hou Li Dynamic depletion of vortex stretching and non-blowup of the 3-D incompressible Euler equations Journal of Nonlinear Science 2006",
 "Necas Ruzicka Sverak On Leray's self-similar solutions of the Navier-Stokes equations Acta Mathematica 1996",
 "Tsai On Leray's self-similar solutions of the Navier-Stokes equations satisfying local energy estimates 1998",
 "Luo Hou Potentially singular solutions of the 3D axisymmetric Euler equations Proceedings of the National Academy of Sciences 2014",
 "Luo Hou Toward the finite-time blowup of the 3D axisymmetric Euler equations a numerical investigation Multiscale Modeling and Simulation 2014",
 "Budd Huang Russell Moving mesh methods for problems with blow-up SIAM Journal on Scientific Computing 1996",
 "Budd Chen Russell New self-similar solutions of the nonlinear Schrodinger equation with moving mesh computations Journal of Computational Physics 1999",
 "Eggers Fontelos The role of self-similarity in singularities of partial differential equations Nonlinearity 2009",
 "Kiselev Sverak Small scale creation for solutions of the incompressible two-dimensional Euler equation Annals of Mathematics 2014",
 "Elgindi Finite-time singularity formation for C1alpha solutions to the incompressible Euler equations Annals of Mathematics 2021",
 "Elgindi Jeong Finite-time singularity formation for strong solutions to the axisymmetric 3D Euler equations Annals of PDE",
]
for s in QUERIES:
    try:
        q(s)
    except Exception as e:
        print("### QUERY:", s, "\n  ERROR", e, "\n")
    time.sleep(0.6)

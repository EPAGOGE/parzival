import urllib.request, json
dois = {
 "ARMA Abe-Ginsberg-Jeong (CLAIMED)":"10.1007/s00205-026-02195-3",
 "SIAM JMA Huang-Qin-Wang (CLAIMED)":"10.1137/24M168845X",
 "AnnPDE Cordoba-MZ-Zheng (CLAIMED)":"10.1007/s40818-025-00214-2",
 "MMS ChenHou PartII (CLAIMED)":"10.1137/23M1580395",
 "PNAS ChenHou 2025 (CLAIMED)":"10.1073/pnas.2500940122",
 "AdvMath Cordoba-LS-MZ (FOUND on arXiv)":"10.1016/j.aim.2025.110480",
}
for label,d in dois.items():
    try:
        req=urllib.request.Request("https://api.crossref.org/works/"+d,
              headers={"User-Agent":"citation-verify/1.0 (mailto:hill.jt@icloud.com)"})
        m=json.load(urllib.request.urlopen(req,timeout=45))["message"]
        au="; ".join((a.get("family","?")+", "+a.get("given","?")[:12]) for a in m.get("author",[])[:6])
        yr=m.get("published",{}).get("date-parts",[[None]])[0][0]
        print(f"[OK]   {label}\n       DOI:     {d}\n       TITLE:   {m.get('title',['?'])[0]}\n       AUTHORS: {au}\n       VENUE:   {(m.get('container-title') or ['?'])[0]}\n       VOL/ISS: vol {m.get('volume','?')} iss {m.get('issue','?')} pages {m.get('page','?')} art {m.get('article-number','?')} year {yr}\n")
    except Exception as ex:
        print(f"[FAIL] {label}\n       DOI:     {d}\n       ERROR:   {type(ex).__name__} {ex}\n")

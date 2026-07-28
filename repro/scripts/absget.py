import json, sys, time, urllib.request
MAIL="hill.jt@icloud.com"
for doi in sys.argv[1:]:
    try:
        req=urllib.request.Request(f"https://api.crossref.org/works/{doi}?mailto={MAIL}",
            headers={"User-Agent":f"ref-verify/1.0 (mailto:{MAIL})"})
        m=json.load(urllib.request.urlopen(req,timeout=60))["message"]
        print("==",doi,"|",(m.get("title") or ["?"])[0][:110])
        ab=m.get("abstract")
        print((" ".join(ab.split())[:1400] if ab else "(no abstract in Crossref)"))
    except Exception as e:
        print("==",doi,"ERROR",e)
    print()
    time.sleep(0.4)

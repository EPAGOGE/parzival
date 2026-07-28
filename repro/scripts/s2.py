import re,html,urllib.parse,subprocess
def q(term,n=10):
    u="http://export.arxiv.org/api/query?search_query="+urllib.parse.quote(term)+"&max_results=%d&sortBy=relevance"%n
    t=subprocess.run(["curl","-sSL",u],capture_output=True,text=True).stdout
    print("\n##### ",term)
    for e in re.findall(r'<entry>(.*?)</entry>', t, re.S):
        i=re.search(r'<id>(.*?)</id>',e,re.S).group(1).split('/abs/')[-1]
        ti=html.unescape(' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split()))
        print(' ',i,'|',ti)
for term in ['abs:"steering vector" AND abs:"retrieval"',
             'abs:"activation steering" AND abs:"memory"',
             'all:"episodic memory" AND all:"cross-attention" AND all:"language model" AND all:"one-shot"']:
    q(term)

import sys,re,html,urllib.parse,subprocess
def q(term,n=12):
    u="http://export.arxiv.org/api/query?search_query="+urllib.parse.quote(term)+"&max_results=%d&sortBy=relevance"%n
    t=subprocess.run(["curl","-sSL",u],capture_output=True,text=True).stdout
    print("\n##### ",term)
    for e in re.findall(r'<entry>(.*?)</entry>', t, re.S):
        i=re.search(r'<id>(.*?)</id>',e,re.S).group(1).split('/abs/')[-1]
        ti=html.unescape(' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split()))
        print(' ',i,'|',ti)
for term in ['all:"preference memory" AND all:"language model"',
             'abs:"affective memory" AND abs:"language model"',
             'all:"one-shot" AND all:"binding" AND all:"key-value memory" AND all:"transformer"',
             'abs:"personalized" AND abs:"cross-attention" AND abs:"user embeddings" AND abs:"LLM"']:
    q(term)

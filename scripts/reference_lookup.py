import urllib.request,urllib.parse,json,concurrent.futures
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'audit'
queries=['CKMDiff channel knowledge map','How much data is needed for channel knowledge map construction','channel knowledge map hybrid beamforming Di Wu']
def fetch(q):
 url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic':q,'rows':5})
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CKM-paper-reference-audit/1.0'}),timeout=40) as r:data=json.load(r)
  return {'query':q,'url':url,'items':[{k:x.get(k) for k in ['DOI','title','author','container-title','published','volume','issue','page','type']} for x in data['message']['items']]}
 except Exception as e:return {'query':q,'error':repr(e)}
with concurrent.futures.ThreadPoolExecutor(3) as pool:result=list(pool.map(fetch,queries))
(out/'reference_lookup.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for r in result:
 print(r['query'])
 for x in r.get('items',[]):print(x['DOI'],x['title'],x['container-title'],x['published'])
 if 'error' in r:print(r['error'])

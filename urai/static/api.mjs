/** JSON client for the service API. The page uses the same-origin form; tests pass the server's base URL. */
export function apiClient(base=''){
 return async function api(path,method='GET',data){
  const r=await fetch(base+'/api/'+path,{method,headers:{'Content-Type':'application/json'},...(data===undefined?{}:{body:JSON.stringify(data)})});
  const value=await r.json();
  if(!r.ok)throw Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail||value));
  return value;
 };
}

//TODO need to refactor this to handle dynamically loaded template for display
const displays: {[key: string]: Function} = {};

export function registerDisplay(name: string, fun: Function){
  displays[name] = fun;
}


export function htmlForType(d: any, key: string): string {
  const metaObj = d[key].content;
  // only show column if it is dataframe
  let header = '';
  const types = d[key].portType as string[][];
  types.forEach((typeNames: string[])=> {
    if (typeNames[0] in displays){
      header = displays[typeNames[0]](metaObj);
    }
  });
  return header;
}

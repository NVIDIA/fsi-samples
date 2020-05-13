export function htmlForType(d: any, key: string): string {
  let header = '';
  // only show column if it is dataframe
  const types = d[key].portType as string[];
  const dfType = types.findIndex(d => {
    return d.indexOf('DataFrame') >= 0;
  });

  if (dfType >= 0) {
    const columnObj = d[key].content;
    const columnKeys = Object.keys(columnObj);
    if (columnKeys.length > 0) {
      header += '<table>';
      header += '<tr>';
      header += '<th>Column Name</th>';
      for (let i = 0; i < columnKeys.length; i++) {
        header += `<th>${columnKeys[i]}</th>`;
      }
      header += '</tr>';
      header += '<tr>';
      header += '<th>Type</th>';
      for (let i = 0; i < columnKeys.length; i++) {
        header += `<td>${columnObj[columnKeys[i]]}</td>`;
      }
      header += '</tr>';
      header += '</table>';
    }
  }

  const nmType = types.findIndex(d => {
    return d.indexOf('NmTensor') >= 0;
  });

  if (nmType >= 0) {
    const columnObj = d[key].content;
    const axes = columnObj['axes'];
    const element = columnObj['element'];
    header += '<table>';
    header += '<tr>';
    header += '<th>Axes: </th>';
    if ('axes' in columnObj && axes.length > 0) {
      for (let i = 0; i < axes.length; i++) {
        if (axes[i]['size']) {
          header += `<th>${i === 0 ? '(' : ''}${axes[i]['kind']}(${
            axes[i]['size']
          })${i === axes.length - 1 ? ')' : ''}</th>`;
        } else {
          header += `<th>${i === 0 ? '(' : ''}${axes[i]['kind']}${
            i === axes.length - 1 ? ')' : ''
          }</th>`;
        }
      }
    } else {
      header += '<th>()</th>';
    }
    header += '</tr>';
    header += '</table>';
    header += '<ul>';
    if ('types' in element) {
      header += `<li>Element Type: ${element['types'][0]}</li>`;
      if ('fileds' in element && element['fields'] !== 'None') {
        header += `<li>Element fileds: ${element['fields']}</li>`;
      }
      if ('parameters' in element && element['parameters'] !== '{}') {
        header += `<li>Element parameters: ${element['parameters']}</li>`;
      }
    }
    header += '</ul>';
  }

  return header;
}

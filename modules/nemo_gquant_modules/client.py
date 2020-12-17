
validation_fun = """
  //first check types

  const reqElement = required['element'];
  const outElement = outputs['element'];
  if (
    outElement['types'][0] !== 'VoidType' &&
    reqElement['types'][0] !== 'VoidType'
  ) {
    if (
      outElement['types'].findIndex(
        (d) => d === reqElement['types'][0]
      ) < 0
    ) {
      // req type should be generic,
      // out type should be specific, i.e. subclass of req
      // first required element type should be the parent type of the output element
      return false;
    }
    if (outElement['fields'] !== reqElement['fields']) {
      return false;
    }
    if (outElement['parameters'] !== reqElement['parameters']) {
      return false;
    }
  }

  const reqAxes = required['axes'];
  const outAxes = outputs['axes'];
  if (reqAxes.length === 0) {
    return true;
  }
  if (reqAxes.length !== outAxes.length) {
    return false;
  }
  for (let i = 0; i < reqAxes.length; i++) {
    if (reqAxes[i]['kind'] !== outAxes[i]['kind']) {
      return false;
    }
    if (
      reqAxes[i]['size'] !== null &&
      outAxes[i]['size'] !== null &&
      reqAxes[i]['size'] !== outAxes['size']
    ) {
      return false;
    }
  }
  return true;
"""

display_fun = """
    const axes = metaObj['axes'];
    const element = metaObj['element'];
    header = '';
    header += '<table>';
    header += '<tr>';
    header += '<th>Axes: </th>';
    if ('axes' in metaObj && axes.length > 0) {
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
    return header;
"""

validation = {}
display = {}
validation['nemo.core.neural_types.neural_type.NmTensor'] = validation_fun
display['nemo.core.neural_types.neural_type.NmTensor'] = display_fun

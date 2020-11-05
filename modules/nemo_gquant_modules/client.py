
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

validation = {}
validation['nemo.core.neural_types.neural_type.NmTensor'] = validation_fun

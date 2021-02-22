
display_fun = """
    const columnKeys = Object.keys(metaObj);
    let header = '';
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
        header += `<td>${metaObj[columnKeys[i]]}</td>`;
      }
      header += '</tr>';
      header += '</table>';
    }
    return header;
"""

validation = {}
display = {}
display['cudf.core.dataframe.DataFrame'] = display_fun
display['dask_cudf.core.DataFrame'] = display_fun
display['pandas.core.frame.DataFrame'] = display_fun
display['dask.dataframe.core.DataFrame'] = display_fun

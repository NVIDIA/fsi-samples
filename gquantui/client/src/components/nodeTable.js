import React from 'react';
import CategoryRow from './categoryRow';
import NodeRow from './nodeRow';

class NodeTable extends React.Component {
    render() {

      const filterText = this.props.filterText; 
      const rows = [];
       for (let k in this.props.allNodes){
          rows.push(
              <CategoryRow category={k} key={k}/>
          );
            this.props.allNodes[k].forEach((d) => {
                if (filterText==='' || d.type.toLowerCase().indexOf(filterText) !== -1) { 
                 rows.push(<NodeRow type={d.type} key={d.id} />);
                }
          })
       }
  
      return (
        <table onClick={this.props.onClick}>
          <thead>
            <tr>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      );
    }
  }

  export default NodeTable;
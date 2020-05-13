import React from 'react';


class NodeRow extends React.Component {
    render() {
      return (
        <tr>
          <td>{this.props.type}</td>
        </tr>
      );
    }
  }

export default NodeRow;
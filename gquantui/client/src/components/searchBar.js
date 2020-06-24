import React from 'react';
import NodeTable from './nodeTable';

class SearchBar extends React.Component {

    constructor(props) {
        super(props);
        this.handleFilterTextChange = this.handleFilterTextChange.bind(this);
        this.state = {filterText: ""};
    }

    handleFilterTextChange(e) {
        this.setState({filterText: e.target.value});
    }

    render() {
      return (
        <div>
            <input type="text" placeholder="Search..."  value={this.state.filterText} onChange={this.handleFilterTextChange.bind(this)}/>
            <NodeTable allNodes={this.props.allNodes} filterText={this.state.filterText} onClick={this.props.handleClicked}/>
        </div>
      );
    }
  }

export default SearchBar;
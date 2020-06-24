import React from 'react';
import SearchBar from './searchBar';
import styled from '@emotion/styled';

class AddNodeMenu extends React.Component {
    constructor(props) {
      super(props);
      this.handleClicked = this.handleClicked.bind(this);
    }

    handleClicked(d) {
        if (d.target.tagName==='TD'){
            let nodeType = d.target.innerText;
            var result = null;
            var found = false;
            for (let k in this.props.allNodes){
                for (let i = 0; i < this.props.allNodes[k].length; i++){
                    if (this.props.allNodes[k][i].type === nodeType){
                        found = true;
                        result = this.props.allNodes[k][i];
                    }
                }
                if (found)break;
            }
            if (found) {
                result['x'] = this.props.nodeX;
                result['y'] = this.props.nodeY;
                result['id'] = Math.random().toString(36).substring(2, 15);
                this.props.currentNodes.push(result);
                this.props.setChartState({'nodes':this.props.currentNodes});
                this.props.setMenuState({'opacity': 0, x: -1000, y:-1000});
            }
        }
     }
  
          //<NodeTable  key="random3" allNodes={this.props.allNodes} filterText={this.state.filterText} onClick={this.handleClicked}/>
    render() {
        let x = (this.props.x + 25) + 'px';
        let y = (this.props.y) + 'px';

        const Menu = styled.div`
            text-align: left;
            padding: 10px;
            position: absolute;
            background-color: yellowgreen;
            opacity: ${this.props.opacity};
            left: ${x};
            top:  ${y};
        `
      return (
        <Menu>
          <SearchBar
          allNodes={this.props.allNodes}
          handleClicked={this.handleClicked}
          />
        </Menu>
      );
    }
  }
  
export default AddNodeMenu;
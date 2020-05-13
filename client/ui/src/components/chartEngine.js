import React from "react";
import * as d3 from "d3";
import * as d4 from "d3-dag";
import Chart from './chart';

class ChartEngine extends React.Component {
  constructor(props) {
    super(props);
    this.state = { nodes: [], edges: [] };
  }

  updateLayout(nodes, edges){
    let connection_info = {};
    for (let i = 0; i < edges.length; i++){
        let children = edges[i].to.split('.')[0];
        let parent = edges[i].from.split('.')[0];
        if  (children in connection_info){
            if (connection_info[children].findIndex((d)=>d===parent) < 0){
              connection_info[children].push(parent);
            }
        }
        else{
          connection_info[children] = [parent];
        }
    }
  
    let data = nodes.map((d)=>{
       if (d['id'] in connection_info){
           d['parentIds'] = connection_info[d['id']];
       }
       return d;
    })
    let dagData = d4.dagStratify()(data);
    d4.sugiyama()
        .size([this.props.height, this.props.width])
        .layering(d4.layeringSimplex())
        .decross(d4.decrossOpt())
        .coord(d4.coordVert())(dagData);
  
    dagData.descendants().forEach((d)=>{d.data['y']=d.x; d.data['x']=d.y; return});
    return data;
  }

layout(nodes, edges){
      let layoutNodes = this.updateLayout(nodes, edges);
      this.setState({ nodes: layoutNodes, edges: edges});
}

componentDidMount() {
    let getData = async (d) => {
      let graph = await d3.json("http://localhost:8787/graph");
      let allNodes = await d3.json("http://localhost:8787/add");
      return [graph, allNodes];
    };
    getData().then((d) => {
      const [graph, allNodes] = d;
      this.setState({allNodes: allNodes});
      this.layout(graph['nodes'], graph['edges']);
    });

  }

  render() {
    return (<Chart nodes={this.state.nodes}
      edges={this.state.edges}
      setChartState={this.setState.bind(this)}
      width={this.props.width}
      height={this.props.height}
      allNodes={this.state.allNodes}
      layout={this.layout.bind(this)} />);
  }
}

export default ChartEngine;

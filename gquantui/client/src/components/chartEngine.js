import React from "react";
import * as d3 from "d3";
import * as d4 from "d3-dag";
import Chart from './chart';
import FileBrowser, {Icons} from 'react-keyed-file-browser';
import Moment from 'moment';
import DetailView from "./detailedView";


class ChartEngine extends React.Component {
  constructor(props) {
    super(props);
    this.state = { nodes: [], edges: [], files: [] };
  }

  updateLayout(nodes, edges, transform){
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
    
    dagData.descendants().forEach((d)=>{
      if (transform){
        let newPosition = transform.invert([d.y, d.x]);
        d.data['y'] = newPosition[1];
        d.data['x'] = newPosition[0]; 
      }
      else{
        d.data['y']=d.x; 
        d.data['x']=d.y; 
      }
      return
    });
    return data;
  }

layout(nodes, edges, transform){
  if (nodes.length === 0) {
    return;
  }
  let layoutNodes = this.updateLayout(nodes, edges, transform);
  this.setState({ nodes: layoutNodes, edges: edges });
}

componentDidMount() {
    let getData = async (d) => {
      //let graph = await d3.json(process.env.REACT_APP_GRAPH_URL);
      let allNodes = await d3.json(process.env.REACT_APP_ALLNODES_URL);
      let workflows = await d3.json(process.env.REACT_APP_WORKFLOWS_URL);
      return [allNodes, workflows];
    };
    getData().then((d) => {
      const [allNodes, workflows] = d;
      let mapedWorkflows = workflows.map((d)=>{
        d['modified'] = Moment(d.modified);
        return d;
      })
      this.setState({allNodes: allNodes, files: mapedWorkflows});
    });

  }

openWorkflow(file){
  this.setState({ filename: file});
  console.log('open', file);
  let payload = JSON.stringify({filename: file});
  const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload
  };
  fetch(process.env.REACT_APP_GRAPH_URL, requestOptions)
      .then(response => response.json())
      .then(graph => {
        this.layout(graph['nodes'], graph['edges'], null);
      });
}

  render() {
    let style = {
      "float": "left",
      "display": "block",
      "width": "20%"
    }
    let style2 = {
      "float": "left",
      "display": "block",
      "width": "80%"
    }
   
    return (<div>
      <div style={style}>
      <FileBrowser
        files={this.state.files}
        icons={Icons.FontAwesome(4)}
        detailRenderer={DetailView}
        detailRendererProps={{ handleOpen: this.openWorkflow.bind(this) }}
      />
      </div>
      <div style={style2}>
      <Chart nodes={this.state.nodes}
        edges={this.state.edges}
        setChartState={this.setState.bind(this)}
        width={this.props.width}
        height={this.props.height}
        allNodes={this.state.allNodes}
        filename={this.state.filename?this.state.filename:"unname.yaml"}
        layout={this.layout.bind(this)} />
      </div>
  </div>
      );
  }
}

export default ChartEngine;

import React from 'react';
import * as d4 from 'd3-dag';

import {
  IEdge,
  INode,
  IAllNodes,
  IChartInput,
  ContentHandler
} from './document';
import { Chart } from './chart';

interface IProps {
  height: number;
  width: number;
  contentHandler: ContentHandler;
}

interface IState {
  nodes: INode[];
  edges: IEdge[];
  allNodes: IAllNodes;
}

export class ChartEngine extends React.Component<IProps, IState> {
  constructor(props: IProps) {
    super(props);
    this.state = {
      nodes: [],
      edges: [],
      allNodes: {}
    };
    props.contentHandler.contentChanged.connect(
      this.contentChangeHandler,
      this
    );
  }

  contentChangeHandler(sender: ContentHandler, inputs: IChartInput): void {
    const layoutNodes = this.updateLayout(inputs.nodes, inputs.edges, null);
    this.setState({
      nodes: layoutNodes,
      edges: inputs.edges,
      allNodes: inputs.allNodes
    });
  }

  updateLayout(nodes: INode[], edges: IEdge[], transform: any): INode[] {
    if (nodes.length === 0) {
      return nodes;
    }
    const connectionInfo: { [key: string]: string[] } = {};
    for (let i = 0; i < edges.length; i++) {
      const children = edges[i].to.split('.')[0];
      const parent = edges[i].from.split('.')[0];
      if (children in connectionInfo) {
        if (connectionInfo[children].findIndex(d => d === parent) < 0) {
          connectionInfo[children].push(parent);
        }
      } else {
        connectionInfo[children] = [parent];
      }
    }

    const data = nodes.map((d: INode) => {
      if (d['id'] in connectionInfo) {
        d['parentIds'] = connectionInfo[d['id']];
      }
      return d;
    });
    const dagData = d4.dagStratify()(data);
    d4
      .sugiyama()
      .size([this.props.height, this.props.width])
      .layering(d4.layeringSimplex())
      .decross(d4.decrossOpt())
      .coord(d4.coordVert())(dagData);

    dagData.descendants().forEach((d: any) => {
      if (transform) {
        const newPosition = transform.invert([d.y, d.x]);
        d.data['y'] = newPosition[1];
        d.data['x'] = newPosition[0];
      } else {
        d.data['y'] = d.x;
        d.data['x'] = d.y;
      }
      return;
    });
    return data;
  }

  layout(nodes: INode[], edges: IEdge[], transform: any): void {
    if (nodes.length === 0) {
      return;
    }
    const layoutNodes = this.updateLayout(nodes, edges, transform);
    this.setState({ nodes: layoutNodes, edges: edges });
  }

  render(): JSX.Element {
    console.log('chart engine render');
    return (
      <Chart
        nodes={this.state.nodes}
        edges={this.state.edges}
        setChartState={this.setState.bind(this)}
        width={this.props.width}
        height={this.props.height}
        allNodes={this.state.allNodes}
        layout={this.layout.bind(this)}
      />
    );
  }
}

// <Chart
//   nodes={this.state.nodes}
//   edges={this.state.edges}
//   setChartState={this.setState.bind(this)}
//   width={this.props.width}
//   height={this.props.height}
//   allNodes={this.state.allNodes}
//   filename={this.state.filename ? this.state.filename : 'unname.yaml'}
//   layout={this.layout.bind(this)}
// />

import React from 'react';
import * as d4 from 'd3-dag';
import YAML from 'yaml';

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

export function exportWorkFlowNodes(nodes: INode[], edges: IEdge[]): INode[] {
  /**
   * get the gqaunt task graph, which is a list of tasks
   */
  const connectionInfo: { [key: string]: { [key: string]: any } } = {};
  for (let i = 0; i < edges.length; i++) {
    const children = edges[i].to.split('.')[0];
    const childrenPort = edges[i].to.split('.')[1];
    if (children in connectionInfo) {
      connectionInfo[children][childrenPort] = edges[i].from;
    } else {
      connectionInfo[children] = {
        [childrenPort]: edges[i].from
      };
    }
  }
  const output: INode[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const element: any = {};
    element['id'] = node.id;
    element['type'] = node.type;
    element['conf'] = node.conf;
    if (node.id in connectionInfo) {
      element['inputs'] = connectionInfo[node.id];
    } else {
      element['inputs'] = {};
    }
    if ('filepath' in node) {
      element['filepath'] = node.filepath;
    }
    output.push(element);
  }
  return output;
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

  componentWillUnmount(): void {
    this.props.contentHandler.contentChanged.disconnect(
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
    const connectionInfoCtoP: { [key: string]: string[] } = {}; // mapping from children to parents
    const connectionInfoPtoC: { [key: string]: string[] } = {}; // mapping from children to parents

    for (let i = 0; i < edges.length; i++) {
      const child = edges[i].to.split('.')[0];
      const parent = edges[i].from.split('.')[0];
      if (child in connectionInfoCtoP) {
        if (connectionInfoCtoP[child].findIndex(d => d === parent) < 0) {
          connectionInfoCtoP[child].push(parent);
        }
      } else {
        connectionInfoCtoP[child] = [parent];
      }
      if (parent in connectionInfoPtoC) {
        if (connectionInfoPtoC[parent].findIndex(d => d === child) < 0) {
          connectionInfoPtoC[parent].push(child);
        }
      } else {
        connectionInfoPtoC[parent] = [child];
      }
    }

    const dataWithParentsId = nodes.map((d: INode) => {
      if (d['id'] in connectionInfoCtoP) {
        d['parentIds'] = connectionInfoCtoP[d['id']];
      } else {
        d['parentIds'] = [];
      }
      return d;
    });

    const data = dataWithParentsId.filter(
      (d: INode) => d.id in connectionInfoCtoP || d.id in connectionInfoPtoC
    );
    const dataIslands = nodes.filter(
      (d: INode) =>
        !(d.id in connectionInfoCtoP) && !(d.id in connectionInfoPtoC)
    );
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
    return [...data, ...dataIslands];
  }

  layout(nodes: INode[], edges: IEdge[], transform: any): void {
    if (nodes.length === 0) {
      return;
    }
    const layoutNodes = this.updateLayout(nodes, edges, transform);
    this.setState({ nodes: layoutNodes, edges: edges });
  }

  updateWorkFlow(state: IState): void {
    this.setState(state);
    if (state.edges && state.nodes) {
      const output = exportWorkFlowNodes(state.nodes, state.edges);
      const yamlText = YAML.stringify(output);
      this.props.contentHandler.update(yamlText);
    }
  }

  render(): JSX.Element {
    console.log('chart engine render');
    return (
      <Chart
        nodes={this.state.nodes}
        edges={this.state.edges}
        setChartState={this.updateWorkFlow.bind(this)}
        width={this.props.width}
        height={this.props.height}
        allNodes={this.state.allNodes}
        layout={this.layout.bind(this)}
      />
    );
  }
}

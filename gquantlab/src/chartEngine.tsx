import React from 'react';
import * as d4 from 'd3-dag';
import YAML from 'yaml';

import { IEdge, INode, ContentHandler, IChartInput } from './document';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { Chart } from './chart';
import { OUTPUT_COLLECTOR } from './mainComponent';

interface IProps {
  height: number;
  width: number;
  contentHandler: ContentHandler;
}

interface IState {
  nodes: INode[];
  edges: IEdge[];
}

export function exportWorkFlowNodes(nodes: INode[], edges: IEdge[]): INode[] {
  const cleanedNodes = nodes.filter((d: INode) => d.id !== OUTPUT_COLLECTOR);
  const cleanedEdges = edges.filter(
    (d: IEdge) => d.to.split('.')[0] !== OUTPUT_COLLECTOR
  );
  /**
   * get the gqaunt task graph, which is a list of tasks
   */
  const connectionInfo: { [key: string]: { [key: string]: any } } = {};
  for (let i = 0; i < cleanedEdges.length; i++) {
    const children = cleanedEdges[i].to.split('.')[0];
    const childrenPort = cleanedEdges[i].to.split('.')[1];
    if (children in connectionInfo) {
      connectionInfo[children][childrenPort] = cleanedEdges[i].from;
    } else {
      connectionInfo[children] = {
        [childrenPort]: cleanedEdges[i].from
      };
    }
  }
  const output: INode[] = [];
  for (let i = 0; i < cleanedNodes.length; i++) {
    const node = cleanedNodes[i];
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
      edges: []
    };
    props.contentHandler.contentChanged.connect(
      this.contentChangeHandler,
      this
    );
    props.contentHandler.chartStateUpdate.connect(
      this.stateUpdateHandler,
      this
    );
  }

  componentWillUnmount(): void {
    this.props.contentHandler.contentChanged.disconnect(
      this.contentChangeHandler,
      this
    );
    this.props.contentHandler.chartStateUpdate.disconnect(
      this.stateUpdateHandler,
      this
    );
  }

  contentChangeHandler(sender: ContentHandler, inputs: IChartInput): void {
    const layoutNodes = this.updateLayout(
      inputs.nodes,
      inputs.edges,
      null,
      inputs.width,
      inputs.height
    );

    // if (
    //   sender.privateCopy &&
    //   sender.privateCopy.get('cache') &&
    //   !sender.privateCopy.get('cache').nodes
    // ) {
    //   this.updateWorkFlow({
    //     nodes: layoutNodes,
    //     edges: inputs.edges
    //   });
    //   return;
    // }

    this.setState({
      nodes: layoutNodes,
      edges: inputs.edges
    });
  }

  stateUpdateHandler(sender: ContentHandler, inputs: IChartInput): void {
    this.setState({
      nodes: inputs.nodes,
      edges: inputs.edges
    });
  }

  updateLayout(
    nodes: INode[],
    edges: IEdge[],
    transform: any,
    width?: number,
    height?: number
  ): INode[] {
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

    const dataWithParentsId = nodes.map((d: INode, i: number) => {
      if (d['id'] in connectionInfoCtoP) {
        d['parentIds'] = connectionInfoCtoP[d['id']];
      } else {
        d['parentIds'] = [];
      }
      d['x'] = i * 10;
      d['y'] = i * 10;
      return d;
    });

    const data = dataWithParentsId.filter(
      (d: INode) => d.id in connectionInfoCtoP || d.id in connectionInfoPtoC
    );
    const dataIslands = nodes.filter(
      (d: INode) =>
        !(d.id in connectionInfoCtoP) && !(d.id in connectionInfoPtoC)
    );
    if (data.length > 0) {
      const dagData = d4.dagStratify()(data);
      d4
        .sugiyama()
        .size([
          height ? height : this.props.height,
          width ? width : this.props.width
        ])
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
    }
    return [...data, ...dataIslands];
  }

  layout(nodes: INode[], edges: IEdge[], transform: any): void {
    if (nodes.length === 0) {
      return;
    }
    const layoutNodes = this.updateLayout(nodes, edges, transform);
    this.setState({ nodes: layoutNodes, edges: edges });
  }

  private _fixOutputCollectorPorts(state: IState): void {
    const index = state.nodes.findIndex(
      (d: INode) => d.id === OUTPUT_COLLECTOR
    );

    if (index < 0) {
      return;
    }
    const connectedEdges = state.edges.filter(
      (d: IEdge) => d.to.split('.')[0] === OUTPUT_COLLECTOR
    );
    const usedPortNames = connectedEdges.map((d: IEdge) => d.to.split('.')[1]);
    const outputCollector = state.nodes[index];
    // total
    const totalNeed = usedPortNames.length + 1;
    // reset the input ports
    outputCollector.inputs = [];
    for (let i = 0; i < totalNeed; i++) {
      outputCollector.inputs.push({ name: `in${i + 1}`, type: ['any'] });
    }
    connectedEdges.forEach((d: IEdge, i: number) => {
      d.to = `${OUTPUT_COLLECTOR}.in${i + 1}`;
    });
    //inputs: [ {name: "in1", type: ['any']}]
  }

  updateWorkFlow(state: IState): void {
    if (state.edges && state.nodes) {
      this._fixOutputCollectorPorts(state);
      const output = exportWorkFlowNodes(state.nodes, state.edges);
      if (this.props.contentHandler.privateCopy) {
        this.props.contentHandler.privateCopy.set('value', output);
        const stateCopy = JSON.parse(JSON.stringify(state));
        this.props.contentHandler.privateCopy.set('cache', stateCopy);
        this.props.contentHandler.privateCopy.save();
        console.log('edges:', state.edges.length, 'nodes:', state.nodes.length);
      }
      const yamlText = YAML.stringify(output);
      this.props.contentHandler.update(yamlText);
    }
    this.setState(state);
  }

  render(): JSX.Element {
    console.log('chart engine render');
    return (
      <Chart
        contentHandler={this.props.contentHandler}
        nodes={this.state.nodes}
        edges={this.state.edges}
        setChartState={this.updateWorkFlow.bind(this)}
        width={this.props.width ? this.props.width : 100}
        height={this.props.height ? this.props.height : 100}
        layout={this.layout.bind(this)}
      />
    );
  }
}

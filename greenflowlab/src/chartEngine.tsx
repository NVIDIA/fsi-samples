import React from 'react';
import { dagStratify, sugiyama, layeringLongestPath, decrossTwoLayer, coordCenter } from 'd3-dag';
//import YAML from 'yaml';
import jsyaml from 'js-yaml';

import { IEdge, INode, ContentHandler, IChartInput } from './document';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { Chart } from './chart';
import { OUTPUT_COLLECTOR } from './mainComponent';

interface IProps {
  contentHandler: ContentHandler;
}

const DefaultWidth = 100;
const DefaultHeight = 100;

function duplicateName(sourceList: INode[], checkName: string): boolean {
  const id = sourceList.findIndex(item => item.id === checkName);
  return id >= 0;
}

const nameReg = /-(\d+)$/;

function changeName(oldname: string): string {
  const matchResult = oldname.match(nameReg);
  if (matchResult) {
    const number = matchResult[1];
    const newNumber = (parseInt(number) + 1).toString();
    const index = matchResult['index'];
    const newName = oldname.substr(0, index) + '-' + newNumber;
    return newName;
  } else {
    return oldname + '-1';
  }
}

export interface IState {
  height: number;
  width: number;
  nodes: INode[];
  edges: IEdge[];
}

/**
 * Serialize the graph to list of INode objects that can be saved into files.
 * UI related infomration is stripped out. The special OUTPUT_COLLECTOR node is removed
 * @param nodes
 * @param edges
 * @returns list of INode that can be serialized
 */

export function exportWorkFlowNodes(nodes: INode[], edges: IEdge[]): INode[] {
  // const cleanedNodes = nodes.filter((d: INode) => d.id !== OUTPUT_COLLECTOR);
  // const cleanedEdges = edges.filter(
  //   (d: IEdge) => d.to.split('.')[0] !== OUTPUT_COLLECTOR
  // );
  const cleanedNodes = nodes;
  const cleanedEdges = edges;
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
    element['id'] = node.id === OUTPUT_COLLECTOR ? '' : node.id;
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
    if ('module' in node) {
      element['module'] = node.module;
    }
    output.push(element);
  }
  return output;
}

export class ChartEngine extends React.Component<IProps, IState> {
  constructor(props: IProps) {
    super(props);
    this.state = {
      height: null,
      width: null,
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
    props.contentHandler.sizeStateUpdate.connect(this.stateHandleResize, this);
    this.props.contentHandler.contentReset.connect(
      this.contentResetHandler,
      this
    );
    this.props.contentHandler.saveCache.connect(this.saveCacheHandler, this);
    this.props.contentHandler.includeContent.connect(
      this.contentIncludeHandler,
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
    this.props.contentHandler.chartStateUpdate.disconnect(
      this.stateHandleResize,
      this
    );
    this.props.contentHandler.contentReset.disconnect(
      this.contentResetHandler,
      this
    );
    this.props.contentHandler.saveCache.disconnect(this.saveCacheHandler, this);
    this.props.contentHandler.includeContent.disconnect(
      this.contentIncludeHandler,
      this
    );
  }

  /**
   *  handle the raw graph nodes and edge changes
   *  recalculate the layout
   * @param sender
   * @param inputs
   */

  contentChangeHandler(sender: ContentHandler, inputs: IChartInput): void {
    if (!inputs.width && !inputs.height) {
      // if the size is not determined, do nothing
      return;
    }
    const layoutNodes = this._updateLayout(
      inputs.nodes,
      inputs.edges,
      null,
      inputs.width ? inputs.width : DefaultWidth,
      inputs.height ? inputs.height : DefaultHeight
    );
    // if the cache is empty and the height/width are determined, populate it so it can be shared between GreenflowViews
    if (
      this.props.contentHandler.privateCopy &&
      this.props.contentHandler.privateCopy.get('cache') &&
      !this.props.contentHandler.privateCopy.get('cache').nodes &&
      inputs.width &&
      inputs.height
    ) {
      console.log('empty cache', this.state);
      const newState = { nodes: layoutNodes, edges: inputs.edges };
      const stateCopy = JSON.parse(JSON.stringify(newState));
      this.props.contentHandler.privateCopy.set('cache', stateCopy);
      this.props.contentHandler.privateCopy.save();
    }
    this.setState({
      nodes: layoutNodes,
      edges: inputs.edges,
      width: inputs.width ? inputs.width : this.state.width,
      height: inputs.height ? inputs.height : this.state.height
    });
  }

  saveCacheHandler(sender: ContentHandler): void {
    if (this.props.contentHandler.privateCopy) {
      const stateCopy = JSON.parse(JSON.stringify(this.state));
      this.props.contentHandler.privateCopy.set('cache', stateCopy);
      this.props.contentHandler.privateCopy.save();
    }
  }

  /**
   *  handle importing raw graph nodes and edges,
   * resolve the name collision,
   *  recalculate the layout, no size information
   * @param sender
   * @param inputs
   */

  contentIncludeHandler(sender: ContentHandler, inputs: IChartInput): void {
    const currentNodes = this.state.nodes;
    const currentEdges = this.state.edges;
    let outputCollectorDup = false;

    inputs.nodes.forEach(d => {
      let name = d.id;
      if (d.id === OUTPUT_COLLECTOR) {
        if (duplicateName(currentNodes, name)) {
          outputCollectorDup = true;
        }
        // do nothing about the output collector
        return;
      }
      while (duplicateName(currentNodes, name)) {
        name = changeName(name);
      }
      if (name !== d.id) {
        // need to clean up the edges
        inputs.edges.forEach(edge => {
          if (edge.from.split('.')[0] === d.id) {
            edge.from = name + '.' + edge.from.split('.')[1];
          }
          if (edge.to.split('.')[0] === d.id) {
            edge.to = name + '.' + edge.to.split('.')[1];
          }
        });
        d.id = name;
      }
    });

    // add nodes
    currentNodes.forEach(d => {
      if (outputCollectorDup && d.id === OUTPUT_COLLECTOR) {
        return;
      }
      inputs.nodes.push(d);
    });

    // add edges
    currentEdges.forEach(d => {
      inputs.edges.push(d);
    });

    this.contentResetHandler(sender, inputs);
  }

  /**
   *  handle the raw graph nodes and edge changes
   *  recalculate the layout, no size information
   * @param sender
   * @param inputs
   */

  contentResetHandler(sender: ContentHandler, inputs: IChartInput): void {
    const layoutNodes = this._updateLayout(
      inputs.nodes,
      inputs.edges,
      null,
      this.state.width,
      this.state.height
    );
    this.updateWorkFlow({
      nodes: layoutNodes,
      edges: inputs.edges,
      height: this.state.height,
      width: this.state.width
    });
  }

  /**
   *
   *  handle the raw graph nodes and edge changes,
   *  just update the nodes and edges, no layout updates
   * @param sender
   * @param inputs
   */
  stateUpdateHandler(sender: ContentHandler, inputs: IChartInput): void {
    this.setState({
      nodes: inputs.nodes,
      edges: inputs.edges
    });
  }

  /**
   *
   *  handle the raw graph nodes and edge changes,
   *  just update the nodes and edges, no layout updates
   * @param sender
   * @param inputs
   */
  stateHandleResize(sender: ContentHandler, inputs: IChartInput): void {
    const layoutNodes = this._updateLayout(
      this.state.nodes,
      this.state.edges,
      null,
      inputs.width,
      inputs.height
    );
    this.setState({
      nodes: layoutNodes,
      edges: this.state.edges,
      width: inputs.width,
      height: inputs.height
    });
  }

  /**
   * Compute the coordinates of the nodes so it fits the give width and height
   * @param nodes
   * @param edges
   * @param transform  transform object returned from d3.zoom
   * @param width if specified, it overrides the state.width
   * @param height
   */
  private _updateLayout(
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
    const layoutNodeID = 'something uniq 24dd8da7-ac82-4b82-aa6b-4d9338ff15b9';
    const layoutNode: INode = {
      id: layoutNodeID,
      width: 0,
      type: 'Empty Layout Node',
      conf: {},
      required: {},
      // eslint-disable-next-line @typescript-eslint/camelcase
      output_meta: [],
      outputs: [],
      schema: {},
      ui: {},
      inputs: []
    };
    const copyNodes = JSON.parse(JSON.stringify(nodes));
    // find all the leaf nodes
    const leaves = copyNodes.filter((d: INode, i: number) => {
      return !(d['id'] in connectionInfoPtoC);
    });
    // connect all the leaf nodes to this layout node
    connectionInfoCtoP[layoutNodeID] = leaves.map((d: INode) => d.id);
    // add this layout node temporally
    copyNodes.push(layoutNode);

    const data = copyNodes.map((d: INode, i: number) => {
      if (d['id'] in connectionInfoCtoP) {
        d['parentIds'] = connectionInfoCtoP[d['id']];
      } else {
        d['parentIds'] = [];
      }
      d['x'] = i * 10;
      d['y'] = i * 10;
      return d;
    });

    const dagData = dagStratify()(data);
      sugiyama()
      .size([height ? height : DefaultHeight, width ? width : DefaultWidth])
      .layering(layeringLongestPath())
      .decross(decrossTwoLayer())
      .coord(coordCenter())(dagData);
    // set the coordinates
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
    const removedLayout = data.filter((d: INode) => d.id !== layoutNodeID);
    return removedLayout;
  }

  /**
   * relayout the nodes and edges
   *
   * @param nodes
   * @param edges
   * @param transform
   */
  layout(nodes: INode[], edges: IEdge[], transform: any): void {
    if (nodes.length === 0) {
      return;
    }
    const layoutNodes = this._updateLayout(
      nodes,
      edges,
      transform,
      this.state.width,
      this.state.height
    );
    this.setState({ nodes: layoutNodes, edges: edges });
  }


  updateWorkFlow(state: IState, update = true): void {
    if (state.edges && state.nodes) {
      //TODO need Need to think about how the logic to handle dynamic ports can be refactored in a generic manner in the future. 
      // Maybe a node could have a self state flag to indicate that it creates ports dynamically.
      const output = exportWorkFlowNodes(state.nodes, state.edges);
      if (this.props.contentHandler.privateCopy) {
        this.props.contentHandler.privateCopy.set('value', output);
        const stateCopy = JSON.parse(JSON.stringify(state));
        this.props.contentHandler.privateCopy.set('cache', stateCopy);
        this.props.contentHandler.privateCopy.save();
        console.log('edges:', state.edges.length, 'nodes:', state.nodes.length);
      }
      const yamlText = jsyaml.safeDump(output);
      this.props.contentHandler.update(yamlText);
    }
    if (update) {
      this.setState(state);
    }
  }

  render(): JSX.Element {
    console.log('chart engine render', this.state.height, this.state.width);
    return (
      <Chart
        contentHandler={this.props.contentHandler}
        nodes={this.state.nodes}
        edges={this.state.edges}
        setChartState={this.updateWorkFlow.bind(this)}
        width={this.state.width ? this.state.width : DefaultWidth}
        height={this.state.height ? this.state.height : DefaultHeight}
        layout={this.layout.bind(this)}
      />
    );
  }
}

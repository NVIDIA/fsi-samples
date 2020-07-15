import React from 'react';
import * as d3 from 'd3';
import { requestAPI } from './gquantlab';
// import YAML from 'yaml';
// import Moment from 'moment';

import {
  handleMouseOver,
  handleMouseOut,
  handleMouseLeft,
  handleClicked,
  handleMouseMoved,
  handleHighlight,
  handleDeHighlight,
  handleEdit,
  handleRightClick
} from './eventHandler';
import { drag } from './dragHandler';
import { handleMouseDown, handleMouseUp } from './connectionHandler';
// import AddNodeMenu from './addNodeMenu';
// import NodeEditor from './nodeEditor';
import { validConnection } from './validator';
import { INode, IAllNodes, IEdge } from './document';
//import styled from '@emotion/styled';

//const Input = styled.input``;
interface IPortInfo {
  [key: string]: any;
}

interface IPoint {
  x: number;
  y: number;
  id: string;
}
interface IMappedEdge {
  source: IPoint;
  target: IPoint;
}

interface IChartState {
  addMenu: boolean;
  x: number;
  y: number;
  opacity: number;
  nodeX: number;
  nodeY: number;
  nodeDatum: any;
}

interface IChartProp {
  nodes: INode[];
  edges: IEdge[];
  setChartState: Function;
  width: number;
  height: number;
  allNodes: IAllNodes;
  layout: Function;
}

export class Chart extends React.Component<IChartProp, IChartState> {
  myRef: React.RefObject<HTMLDivElement>;
  mouse: any;
  mousePage: any;
  starting: any;
  tooltip: any;
  textHeight: number;
  circleHeight: number;
  circleRadius: number;
  bars: any;
  svg: any;
  transform: any;
  link: any;
  mouseLink: any;
  g: any;
  inputPorts: Set<string>;
  outputPorts: Set<string>;
  inputRequriements: { [key: string]: any };
  outputColumns: { [key: string]: any };
  portTypes: { [key: string]: string[] };

  constructor(props: IChartProp) {
    super(props);
    this.myRef = React.createRef<HTMLDivElement>();
    this.mouse = null;
    this.mousePage = null;
    this.starting = null;
    this.tooltip = null;
    this.textHeight = 25;
    this.circleHeight = 20;
    this.circleRadius = 8;
    this.bars = null;
    this.svg = null;
    this.transform = null;
    this.link = null;
    this.mouseLink = null;
    this.g = null;
    this.state = {
      addMenu: true,
      x: -1000,
      y: -1000,
      opacity: 0,
      nodeX: 0,
      nodeY: 0,
      nodeDatum: null
    };
    this.inputPorts = new Set();
    this.outputPorts = new Set();
    this.inputRequriements = {};
    this.outputColumns = {};
    this.portTypes = {};
  }

  portMap(): void {
    this.inputPorts = new Set();
    this.outputPorts = new Set();
    this.inputRequriements = {};
    this.outputColumns = {};
    this.props.nodes.forEach((d: INode) => {
      const nodeId = d.id;
      d.inputs.forEach(k => {
        this.inputPorts.add(nodeId + '.' + k.name);
        this.portTypes[nodeId + '.' + k.name] = k.type;
      });
      d.outputs.forEach(k => {
        this.outputPorts.add(nodeId + '.' + k.name);
        this.portTypes[nodeId + '.' + k.name] = k.type;
      });
      let keys = Object.keys(d.required);
      keys.forEach(k => {
        this.inputRequriements[nodeId + '.' + k] = d.required[k];
      });
      keys = Object.keys(d.output_columns);
      keys.forEach(k => {
        this.outputColumns[nodeId + '.' + k] = d.output_columns[k];
      });
    });
  }

  translateCorr(portStr: string, from = false): IPoint {
    const splits = portStr.split('.');
    const nodeId = splits[0];
    const outputPort = splits[1];
    const nodeObj = this.props.nodes.filter(d => d.id === nodeId)[0];
    if (from) {
      const index = nodeObj.outputs.findIndex(d => d.name === outputPort);
      const x = nodeObj.x + nodeObj.width;
      const y = nodeObj.y + (index + 0.4) * this.circleHeight + this.textHeight;
      const point = { x: x, y: y, id: nodeId };
      return point;
    } else {
      const index = nodeObj.inputs.findIndex(d => d.name === outputPort);
      const x = nodeObj.x;
      const y = nodeObj.y + (index + 0.4) * this.circleHeight + this.textHeight;
      const point = { x: x, y: y, id: nodeId };
      return point;
    }
  }

  edgeMap(d: IEdge): IMappedEdge {
    const sourcePoint = this.translateCorr(d.from, true);
    const targetPoint = this.translateCorr(d.to, false);
    return { source: sourcePoint, target: targetPoint };
  }

  edgeData(): IMappedEdge[] {
    return this.props.edges.map(this.edgeMap.bind(this));
  }

  componentDidMount(): void {
    this.tooltip = d3
      .select(this.myRef.current)
      .append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute');

    const zoom = d3
      .zoom()
      .scaleExtent([0.1, 30])
      .on('zoom', d => {
        this.g.attr('transform', d3.event.transform);
        this.transform = d3.event.transform;
      });

    this.svg = d3
      .select(this.myRef.current)
      .append('svg')
      .attr('width', this.props.width + 200)
      .attr('height', this.props.height + 200)
      .attr(
        'viewBox',
        `0 0 ${this.props.width + 300} ${this.props.height + 300}`
      )
      .attr('font-family', 'sans-serif')
      .attr('font-size', '14')
      .attr('text-anchor', 'end')
      .style('border-style', 'solid')
      .on('mouseleave', handleMouseLeft(this))
      .on('mousemove', handleMouseMoved(this))
      .on('click', handleClicked(this))
      .on('contextmenu', handleRightClick(this))
      .call(zoom.bind(this));

    //d3.select("body").on("keydown", handleKey(this));
    this.g = this.svg.append('g');

    this.bars = this.g.selectAll('g').data(this.props.nodes);

    this.link = this.g
      .append('g')
      .attr('stroke', '#999')
      .selectAll('line')
      .data(this.edgeData())
      .join('line');

    this.mouseLink = this.g
      .append('g')
      .attr('stroke', 'red')
      .selectAll('line');
  }

  drawCircles(): void {
    const portsInput = this.bars
      .selectAll('g')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [d])
      .join('g')
      .attr(
        'transform',
        (d: INode, i: number) => `translate(0, ${this.textHeight})`
      )
      .attr('group', 'inputs');

    portsInput
      .selectAll('circle')
      .data((d: INode) => {
        const data = [];
        for (let i = 0; i < d.inputs.length; i++) {
          if (d.inputs[i].name in d.required) {
            const portInfo: IPortInfo = {};
            portInfo['content'] = d.required[d.inputs[i].name];
            portInfo['portType'] = d.inputs[i].type;
            data.push({
              [d.id + '.' + d.inputs[i].name]: portInfo
            });
          } else {
            const portInfo: IPortInfo = {};
            portInfo['content'] = {};
            portInfo['portType'] = d.inputs[i].type;
            data.push({
              [d.id + '.' + d.inputs[i].name]: portInfo
            });
          }
        }
        return data;
      })
      .join('circle')
      .attr('fill', (d: any) => {
        if (!this.starting) {
          return 'blue';
        }
        const key = Object.keys(d)[0];
        if (validConnection(this)(this.starting.from, key)) {
          return 'blue';
        } else {
          return 'white';
        }
      })
      .attr('cx', 0)
      .attr('cy', (d: any, i: number) => (i + 0.4) * this.circleHeight)
      .attr('r', this.circleRadius)
      .on('mouseover', handleMouseOver(this))
      .on('mouseout', handleMouseOut(this))
      .on('mousedown', handleMouseDown(this))
      .on('mouseup', handleMouseUp(this));

    const portsOutput = this.bars
      .selectAll('g')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [d])
      .join('g')
      .attr(
        'transform',
        (d: INode, i: number) => `translate(${d.width}, ${this.textHeight})`
      )
      .attr('group', 'outputs');

    portsOutput
      .selectAll('circle')
      .data((d: INode) => {
        const data = [];
        for (let i = 0; i < d.outputs.length; i++) {
          if (d.outputs[i].name in d.output_columns) {
            const portInfo: IPortInfo = {};
            portInfo['content'] = d.output_columns[d.outputs[i].name];
            portInfo['portType'] = d.outputs[i].type;
            data.push({
              [d.id + '.' + d.outputs[i].name]: portInfo
            });
          } else {
            const portInfo: IPortInfo = {};
            portInfo['content'] = {};
            portInfo['portType'] = d.outputs[i].type;
            data.push({
              [d.id + '.' + d.outputs[i].name]: portInfo
            });
          }
        }
        return data;
      })
      .join('circle')
      .attr('fill', (d: any) => {
        if (!this.starting) {
          return 'green';
        }
        const key = Object.keys(d)[0];
        if (validConnection(this)(this.starting.from, key)) {
          return 'green';
        } else {
          return 'white';
        }
      })
      .attr('cx', 0)
      .attr('cy', (d: any, i: number) => (i + 0.4) * this.circleHeight)
      .attr('r', this.circleRadius)
      .on('mouseover', handleMouseOver(this))
      .on('mouseout', handleMouseOut(this))
      .on('mousedown', handleMouseDown(this))
      .on('mouseup', handleMouseUp(this));
  }

  drawLinks(): void {
    this.link = this.link
      .data(this.edgeData())
      .join('line')
      .attr('x1', (d: IMappedEdge) => d.source.x)
      .attr('y1', (d: IMappedEdge) => d.source.y)
      .attr('x2', (d: IMappedEdge) => d.target.x)
      .attr('y2', (d: IMappedEdge) => d.target.y);
  }

  componentDidUpdate() {
    this.bars = this.bars
      .data(this.props.nodes)
      .join('g')
      .attr('transform', (d: INode, i: number) => `translate(${d.x}, ${d.y})`);

    this.bars
      .selectAll('rect')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [
        {
          w: d.width,
          h:
            this.textHeight +
            Math.max(d.inputs.length, d.outputs.length) * this.circleHeight
        }
      ])
      .join('rect')
      .attr('fill', 'steelblue')
      .attr('width', (d: { w: number; h: number }) => d.w)
      .attr('height', (d: { w: number; h: number }) => d.h)
      .on('mouseover', handleHighlight(this, 'red', 'pointer'))
      .on('mouseout', handleDeHighlight(this))
      .on('click', handleEdit(this));

    this.bars
      .selectAll('rect')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [{ w: d.width, h: this.textHeight }])
      .join('rect')
      .attr('fill', 'seagreen')
      .attr('width', (d: { w: number; h: number }) => d.w)
      .attr('height', (d: { w: number; h: number }) => d.h)
      .call(drag(this.props.setChartState, this))
      .on('mouseover', handleHighlight(this, 'black', 'grab'))
      .on('mouseout', handleDeHighlight(this));

    this.bars
      .selectAll('text')
      .filter((d: INode, i: number) => i === 0)
      .data((d: INode, i: number) => [{ w: d.width, id: d.id }])
      .join('text')
      .attr('fill', 'white')
      .attr('x', (d: { w: number; id: string }) => d.w)
      .attr('y', 0)
      .attr('dy', '1.00em')
      .attr('dx', '-1.00em')
      .text((d: { w: number; id: string }) => d.id)
      .call(drag(this.props.setChartState, this))
      .on('mouseover', handleHighlight(this, 'black', 'grab'))
      .on('mouseout', handleDeHighlight(this));

    this.bars
      .selectAll('text')
      .filter((d: INode, i: number) => i === 1)
      .data((d: INode, i: number) => [{ w: d.width, text: d.type }])
      .join('text')
      .attr(
        'transform',
        (d: { w: number; text: string }, i: number) =>
          `translate(0, ${this.textHeight})`
      )
      .attr('fill', 'black')
      .attr('x', (d: { w: number; text: string }, i: number) => d.w)
      .attr('y', 0)
      .attr('dy', '1.00em')
      .attr('dx', '-1.00em')
      .text((d: { w: number; text: string }, i: number) => d.text)
      .on('mouseover', handleHighlight(this, 'red', 'pointer'))
      .on('mouseout', handleDeHighlight(this))
      .on('click', handleEdit(this));

    this.drawCircles();
    this.drawLinks();
  }

  reLayout(): void {
    this.props.layout(this.props.nodes, this.props.edges, this.transform);
  }

  updateInputs(json: string): void {
    /**
     * send the taskgraph to backend to run the column-flow logics so all the output types and names are computed
     */
    const workflows = requestAPI<any>('load_graph', {
      body: json,
      method: 'POST'
    });

    workflows.then((data: any) => {
      const newNode: {
        [key: string]: { required: any; outputColumns: any };
      } = {};
      data.nodes.forEach((d: INode) => {
        newNode[d.id] = {
          required: d.required,
          outputColumns: d.output_columns
        };
      });
      this.props.nodes.forEach((d: INode) => {
        if (d.id in newNode) {
          d.required = newNode[d.id].required;
          // eslint-disable-next-line @typescript-eslint/camelcase
          d.output_columns = newNode[d.id].outputColumns;
        }
      });
      this.props.setChartState({
        nodes: this.props.nodes,
        edges: this.props.edges
      });
    });
  }

  configFile(): INode[] {
    /**
     * get the gqaunt task graph, which is a list of tasks
     */
    const connectionInfo: { [key: string]: { [key: string]: any } } = {};
    for (let i = 0; i < this.props.edges.length; i++) {
      const children = this.props.edges[i].to.split('.')[0];
      const childrenPort = this.props.edges[i].to.split('.')[1];
      if (children in connectionInfo) {
        connectionInfo[children][childrenPort] = this.props.edges[i].from;
      } else {
        connectionInfo[children] = {
          [childrenPort]: this.props.edges[i].from
        };
      }
    }
    const output: INode[] = [];
    for (let i = 0; i < this.props.nodes.length; i++) {
      const node = this.props.nodes[i];
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

  // handleNameChange(event) {
  //   this.props.setChartState({ filename: event.target.value });
  // }

  // downloadConf() {
  //   let output = this.configFile();
  //   let jsonString = JSON.stringify(output);
  //   this.updateInputs(jsonString);
  //   let yamlText = YAML.stringify(output);
  //   const element = document.createElement('a');
  //   const file = new Blob([yamlText], { type: 'text/plain' });
  //   element.href = URL.createObjectURL(file);
  //   element.download = this.props.filename;
  //   document.body.appendChild(element); // Required for this to work in FireFox
  //   element.click();
  // }

  // saveConf() {
  //   let output = this.configFile();
  //   let jsonString = JSON.stringify(output);
  //   this.updateInputs(jsonString);
  //   let yamlText = YAML.stringify(output);

  //   let payload = { filename: this.props.filename, content: yamlText };

  //   const requestOptions = {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify(payload)
  //   };
  //   fetch(process.env.REACT_APP_SAVE_URL, requestOptions)
  //     .then(response => response.json())
  //     .then(workflows => {
  //       let mapedWorkflows = workflows.map(d => {
  //         d['modified'] = Moment(d.modified);
  //         return d;
  //       });
  //       this.props.setChartState({ files: mapedWorkflows });
  //     });
  // }

  //   render() {
  //     this.portMap();
  //     console.log('rendering');
  //     if (this.state.addMenu) {
  //       return (
  //         <div ref={this.myRef}>
  //           {this.props.allNodes && (
  //             <AddNodeMenu
  //               allNodes={this.props.allNodes}
  //               x={this.state.x}
  //               y={this.state.y}
  //               nodeX={this.state.nodeX}
  //               nodeY={this.state.nodeY}
  //               opacity={this.state.opacity}
  //               setChartState={this.props.setChartState}
  //               currentNodes={this.props.nodes}
  //               setMenuState={this.setState.bind(this)}
  //             />
  //           )}
  //           <div>
  //             <Input
  //               type="text"
  //               value={this.props.filename}
  //               onChange={this.handleNameChange.bind(this)}
  //             />
  //             <button onClick={this.reLayout.bind(this)}>Auto Layout</button>
  //             <button onClick={this.downloadConf.bind(this)}>Download</button>
  //             <button onClick={this.saveConf.bind(this)}>Save</button>
  //           </div>
  //         </div>
  //       );
  //     } else {
  //       return (
  //         <div ref={this.myRef}>
  //           <NodeEditor
  //             x={this.state.x}
  //             y={this.state.y}
  //             nodeX={this.state.nodeX}
  //             nodeY={this.state.nodeY}
  //             opacity={this.state.opacity}
  //             nodeDatum={this.state.nodeDatum}
  //             setChartState={this.props.setChartState}
  //             nodes={this.props.nodes}
  //             edges={this.props.edges}
  //             setMenuState={this.setState.bind(this)}
  //           />
  //           <div>
  //             <Input
  //               type="text"
  //               value={this.props.filename}
  //               onChange={this.handleNameChange.bind(this)}
  //             />
  //             <button onClick={this.reLayout.bind(this)}>Auto Layout</button>
  //             <button onClick={this.downloadConf.bind(this)}>Download</button>
  //             <button onClick={this.saveConf.bind(this)}>Save</button>
  //           </div>
  //         </div>
  //       );
  //     }
  //   }
  // }
  //

  render(): JSX.Element {
    this.portMap();
    console.log('rendering');
    if (this.state.addMenu) {
      return <div ref={this.myRef} />;
    }
  }
}

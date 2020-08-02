import React from 'react';
import styled from '@emotion/styled';
import Form from '@rjsf/core';
import 'bootstrap/dist/css/bootstrap.css';
import '../style/editor.css';
// import 'glyphicons-halflings/css/bootstrap-glyphicons';
//import 'glyphicons-halflings/css/glyphicons-halflings.css';
import { INode, IEdge } from './document';
import { OUTPUT_COLLECTOR } from './mainComponent';
import Draggable from 'react-draggable';
//import '../style/bootstrap-glyphicons.css';
interface IProp {
  x: number;
  y: number;
  opacity: number;
  nodeDatum: any;
  setChartState: Function;
  nodes: INode[];
  edges: IEdge[];
  setMenuState: Function;
}

class NodeEditor extends React.Component<IProp> {
  myRef: React.RefObject<any>;
  constructor(props: IProp) {
    super(props);
    this.handleSave = this.handleSave.bind(this);
    this.handleDelete = this.handleDelete.bind(this);
    //    this.myRef = React.createRef();
    this.myRef = React.createRef<any>();
  }

  handleSave(d: any): void {
    const id = this.props.nodeDatum.id;
    const newNodes = this.props.nodes.filter(d => d.id !== id);
    const nodeName = this.myRef.current.value;
    //let nodeConf = d.currentTarget.parentElement.getElementsByTagName('textarea')[0].value;
    if (newNodes.findIndex(d => d.id === nodeName) >= 0) {
      window.alert(`the node id ${nodeName} is not unique`);
      return;
    }
    this.props.nodeDatum.id = nodeName;
    this.props.nodeDatum.conf = d.formData;
    const newEdges = this.props.edges.map(d => {
      let oldNodeName = d.from.split('.')[0];
      if (oldNodeName === id) {
        d.from = nodeName + '.' + d.from.split('.')[1];
      }
      oldNodeName = d.to.split('.')[0];
      if (oldNodeName === id) {
        d.to = nodeName + '.' + d.to.split('.')[1];
      }
      return d;
    });
    this.props.setChartState({ nodes: this.props.nodes, edges: newEdges });
    this.props.setMenuState({ opacity: 0, x: -1000, y: -1000 });
  }

  handleDelete(): void {
    const id = this.props.nodeDatum.id;
    const newNodes = this.props.nodes.filter(d => d.id !== id);
    const newEdges = this.props.edges.filter(d => {
      return d.from.split('.')[0] !== id && d.to.split('.')[0] !== id;
    });
    this.props.setChartState({ nodes: newNodes, edges: newEdges });
    this.props.setMenuState({ opacity: 0, x: -1000, y: -1000 });
  }

  render(): JSX.Element {
    const x = this.props.x + 25 + 'px';
    const y = this.props.y + 'px';

    const Editor = styled.div`
      text-align: left;
      padding: 10px;
      position: fixed;
      background-color: yellowgreen;
      opacity: ${this.props.opacity};
      left: ${x};
      top: ${y};
      z-index: 1000;
    `;
    const Button = styled.button`
      background-color: red;
    `;
    // console.log(this.props.nodeDatum);

    if (this.props.nodeDatum.id === OUTPUT_COLLECTOR) {
      return (
        <Draggable>
          <Editor>
            <Button onClick={this.handleDelete}>Delete</Button>
          </Editor>
        </Draggable>
      );
    }
    return (
      <Draggable>
        <Editor>
          <div>
            <span>Node id:</span>
            <input
              type="text"
              placeholder="unique node name"
              defaultValue={this.props.nodeDatum.id}
              ref={this.myRef}
            />
          </div>
          <Form
            schema={this.props.nodeDatum.schema}
            formData={this.props.nodeDatum.conf}
            uiSchema={this.props.nodeDatum.ui}
            onSubmit={this.handleSave}
          />
          <Button onClick={this.handleDelete}>Delete</Button>
        </Editor>
      </Draggable>
    );
  }
}

export default NodeEditor;

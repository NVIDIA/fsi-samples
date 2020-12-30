import React from 'react';
import styled from '@emotion/styled';
import Form from '@rjsf/core';
import 'bootstrap/dist/css/bootstrap.css';
import '../style/editor.css';
import { INode, IEdge, ContentHandler } from './document';
import { InputDialog } from '@jupyterlab/apputils';
import { OUTPUT_COLLECTOR } from './mainComponent';
import {
  FileSelector,
  PathSelector,
  TaskgraphSelector,
  CsvFileSelector
} from './FilePathSelector';
import { COMMAND_CREATE_CUST_NODE } from './commands';

export interface IEditorProp {
  handler: ContentHandler;
  nodeDatum: any;
  setChartState: Function;
  nodes: INode[];
  edges: IEdge[];
}

class NodeEditor extends React.Component<IEditorProp> {
  myRef: React.RefObject<any>;
  constructor(props: IEditorProp) {
    super(props);
    this.handleSave = this.handleSave.bind(this);
    this.handleDelete = this.handleDelete.bind(this);
    this.handleClone = this.handleClone.bind(this);
    this.handleConvert = this.handleConvert.bind(this);
    this.myRef = React.createRef<any>();
  }

  _bugWorkaround(json: any): void {
    // currently, this bug is not fixed yet, as a work aournd
    // https://github.com/rjsf-team/react-jsonschema-form/issues/1773
    const iterate = (obj: any): void => {
      Object.keys(obj).forEach(key => {
        //console.log(`key: ${key}, value: ${obj[key]}`);
        if (key === 'indicators') {
          const arr = obj[key];
          if (Array.isArray(arr)) {
            arr.forEach(item => {
              console.log(item);
              let counter = 0;
              let large = -1;
              Object.keys(item).forEach(key => {
                if (key.startsWith('function.')) {
                  counter += 1;
                  const id = parseInt(key.split('.')[1]);
                  if (large < id) {
                    large = id;
                  }
                }
              });
              if (counter > 0) {
                console.log('found dup');
                // found the dup
                Object.keys(item).forEach(key => {
                  const id = parseInt(key.split('.')[1]);
                  if (id !== large) {
                    delete item[key];
                  }
                });
              }
            });
          }
        }

        if (typeof obj[key] === 'object') {
          iterate(obj[key]);
        }
      });
    };
    iterate(json);
  }

  handleSave(d: any): void {
    const id = this.props.nodeDatum.id;
    const newNodes = this.props.nodes.filter(node => node.id !== id);
    const nodeName = this.myRef.current.value;
    //let nodeConf = d.currentTarget.parentElement.getElementsByTagName('textarea')[0].value;
    if (newNodes.findIndex(node => node.id === nodeName) >= 0) {
      window.alert(`the node id ${nodeName} is not unique`);
      return;
    }
    // this.props.nodeDatum.id = nodeName;
    // this.props.nodeDatum.conf = d.formData;
    const newEdges = this.props.edges.map(edge => {
      let oldNodeName = edge.from.split('.')[0];
      if (oldNodeName === id) {
        edge.from = nodeName + '.' + edge.from.split('.')[1];
      }
      oldNodeName = edge.to.split('.')[0];
      if (oldNodeName === id) {
        edge.to = nodeName + '.' + edge.to.split('.')[1];
      }
      return edge;
    });

    const outNewNodes = this.props.nodes.map(node => {
      if (node.id === id) {
        node.id = nodeName;
        node.conf = d.formData;
      }
      return node;
    });
    this.props.setChartState({ nodes: outNewNodes, edges: newEdges });
    // this.props.handler.updateEditor.emit({
    //   nodes: [],
    //   nodeDatum: {},
    //   edges: [],
    //   setChartState: null,
    //   handler: this.props.handler
    // });
  }

  handleDelete(): void {
    const id = this.props.nodeDatum.id;
    const newNodes = this.props.nodes.filter(d => d.id !== id);
    const newEdges = this.props.edges.filter(d => {
      return d.from.split('.')[0] !== id && d.to.split('.')[0] !== id;
    });
    this.props.setChartState({ nodes: newNodes, edges: newEdges });
    //    this.props.setMenuState({ opacity: 0, x: -1000, y: -1000 });
    this.props.handler.updateEditor.emit({
      nodes: [],
      nodeDatum: {},
      edges: [],
      setChartState: null,
      handler: this.props.handler
    });
  }

  handleClone(): void {
    const newNode = JSON.parse(JSON.stringify(this.props.nodeDatum));
    newNode.id = Math.random()
      .toString(36)
      .substring(2, 15);
    newNode.x += 20;
    newNode.y += 20;
    this.props.nodes.push(newNode);
    this.props.setChartState({
      nodes: this.props.nodes,
      edges: this.props.edges
    });
  }

  handleConvert(): void {
    // Request a text
    InputDialog.getText({ title: 'Provide a customized Node name' }).then(
      value => {
        console.log('text ' + value.value);
        this.props.handler.commandRegistry.execute(COMMAND_CREATE_CUST_NODE, {
          nodeName: value.value,
          conf: this.props.nodeDatum.conf
        });
      }
    );
  }

  render(): JSX.Element {
    const widgets = {
      FileSelector: FileSelector,
      PathSelector: PathSelector,
      TaskgraphSelector: TaskgraphSelector,
      CsvFileSelector: CsvFileSelector
    };
    const Editor = styled.div`
      text-align: left;
      padding: 10px;
      background-color: yellowgreen;
      overflow-y: auto;
      height: 100%;
    `;
    // const Button = styled.button`
    //   background-color: red;
    // `;
    // console.log(this.props.nodeDatum);
    if (this.props.setChartState === null) {
      return (
        <Editor>
          <h2>Click on the Task Node to Edit</h2>
        </Editor>
      );
    }

    if (this.props.nodeDatum.id === OUTPUT_COLLECTOR) {
      return (
        <Editor>
          <button className={'btn btn-danger'} onClick={this.handleDelete}>
            Delete
          </button>
        </Editor>
      );
    }
    if (this.props.nodeDatum.type === 'CompositeNode') {
      return (
        //     <Draggable>
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
            widgets={widgets}
            formContext={this.props.handler}
          />
          <button className={'btn btn-primary'} onClick={this.handleConvert}>
            Convert it To Node
          </button>
          <button className={'btn btn-primary'} onClick={this.handleClone}>
            Clone
          </button>
          <button className={'btn btn-danger'} onClick={this.handleDelete}>
            Delete
          </button>
        </Editor>
      );
    }
    return (
      //     <Draggable>
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
          widgets={widgets}
          formContext={this.props.handler}
        />
        <button className={'btn btn-primary'} onClick={this.handleClone}>
          Clone
        </button>
        <button className={'btn btn-danger'} onClick={this.handleDelete}>
          Delete
        </button>
      </Editor>
    );
  }
}

export default NodeEditor;

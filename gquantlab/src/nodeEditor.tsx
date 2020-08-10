import React from 'react';
import styled from '@emotion/styled';
import Form from '@rjsf/core';
import 'bootstrap/dist/css/bootstrap.css';
import '../style/editor.css';
import { INode, IEdge, ContentHandler } from './document';
import { OUTPUT_COLLECTOR } from './mainComponent';
import {
  FileSelector,
  PathSelector,
  TaskgraphSelector,
  CsvFileSelector
} from './FilePathSelector';

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
    const Button = styled.button`
      background-color: red;
    `;
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
          <Button onClick={this.handleDelete}>Delete</Button>
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
        <Button onClick={this.handleDelete}>Delete</Button>
      </Editor>
    );
  }
}

export default NodeEditor;

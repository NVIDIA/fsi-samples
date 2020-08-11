// eslint-disable-next-line @typescript-eslint/no-unused-vars
//import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import { WidgetProps } from '@rjsf/core';
import styled from '@emotion/styled';
//import { UseSignal } from '@jupyterlab/apputils';
//import { ContentHandler, IChartInput } from './document';
//import { ContentHandler } from './document';
//import { Widget } from '@lumino/widgets';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
// import { ChartEngine } from './chartEngine';
// import { Message } from '@lumino/messaging';
//import NodeEditor from './nodeEditor';
//import { IEditorProp } from './nodeEditor';

function uuidv4(): string {
  return Math.random()
    .toString(36)
    .substring(2, 15);
}

export class TaskgraphSelector extends React.Component<WidgetProps> {
  private _id: string;
  constructor(props: any) {
    super(props);
  }

  render(): JSX.Element {
    const Input = styled.input`
      width: 100%;
    `;
    const Group = styled.div`
      width: 100%;
    `;
    this._id = 'taskgraph_editor_"' + uuidv4();
    return (
      <Group>
        <Input
          id={this._id}
          type="text"
          className="custom"
          value={this.props.value}
          required={this.props.required}
          onChange={(event): any => {
            return this.props.onChange(event.target.value);
          }}
          onClick={(event): any => {
            if (this.props.formContext.commandRegistry) {
              const path = this.props.formContext.commandRegistry.execute(
                'gquant:selectTheFile',
                { filter: ['.gq.yaml'] }
              );
              path.then(
                (d: any): void => {
                  if (d) {
                    //event.currentTarget.value = d.path;
                    this.props.onChange(d.path);
                  }
                  // eslint-disable-next-line prettier/prettier
                }
              );
            }
          }}
        />
        <button
          onClick={(): void => {
            const inputEle: HTMLInputElement = document.getElementById(
              this._id
            ) as HTMLInputElement;
            if (this.props.formContext.commandRegistry) {
              this.props.formContext.commandRegistry.execute(
                'docmanager:open',
                { path: inputEle.value }
              );
            }
          }}
        >
          Show Taskgraph
        </button>
      </Group>
    );
  }
}

export class CsvFileSelector extends React.Component<WidgetProps> {
  constructor(props: any) {
    super(props);
  }

  render(): JSX.Element {
    const Input = styled.input`
      width: 100%;
    `;

    return (
      <Input
        type="text"
        className="custom"
        value={this.props.value}
        required={this.props.required}
        onChange={(event): any => {
          return this.props.onChange(event.target.value);
        }}
        onClick={(event): any => {
          if (this.props.formContext.commandRegistry) {
            const path = this.props.formContext.commandRegistry.execute(
              'gquant:selectTheFile',
              { filter: ['.csv', '.csv.gz'] }
            );
            path.then(
              (d: any): void => {
                if (d) {
                  //event.currentTarget.value = d.path;
                  this.props.onChange(d.path);
                }
                // eslint-disable-next-line prettier/prettier
            }
            );
          }
        }}
      />
    );
  }
}

export class FileSelector extends React.Component<WidgetProps> {
  constructor(props: any) {
    super(props);
  }

  render(): JSX.Element {
    const Input = styled.input`
      width: 100%;
    `;

    return (
      <Input
        type="text"
        className="custom"
        value={this.props.value}
        required={this.props.required}
        onChange={(event): any => {
          return this.props.onChange(event.target.value);
        }}
        onClick={(event): any => {
          if (this.props.formContext.commandRegistry) {
            const path = this.props.formContext.commandRegistry.execute(
              'gquant:selectTheFile'
            );
            path.then(
              (d: any): void => {
                if (d) {
                  this.props.onChange(d.path);
                }
                // eslint-disable-next-line prettier/prettier
            }
            );
          }
        }}
      />
    );
  }
}

export class PathSelector extends React.Component<WidgetProps> {
  constructor(props: any) {
    super(props);
  }

  render(): JSX.Element {
    const Input = styled.input`
      width: 100%;
    `;
    return (
      <Input
        type="text"
        className="custom"
        value={this.props.value}
        required={this.props.required}
        onChange={(event): any => {
          return this.props.onChange(event.target.value);
        }}
        onClick={(event): any => {
          console.log('event');
          console.log(this.props);
          if (this.props.formContext.commandRegistry) {
            const path = this.props.formContext.commandRegistry.execute(
              'gquant:selectThePath'
            );
            path.then(
              (d: any): void => {
                if (d) {
                  this.props.onChange(d.path);
                }
              }
            );
          }
        }}
      />
    );
  }
}

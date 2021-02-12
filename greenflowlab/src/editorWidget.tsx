// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
import { UseSignal } from '@jupyterlab/apputils';
//import { ContentHandler, IChartInput } from './document';
import { ContentHandler } from './document';
//import { Widget } from '@lumino/widgets';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
// import { ChartEngine } from './chartEngine';
// import { Message } from '@lumino/messaging';
import NodeEditor from './nodeEditor';
import { IEditorProp } from './nodeEditor';

export class EditorView extends ReactWidget {
  private _contentHandler: ContentHandler;
  firstTimeReSize: boolean;

  // onAfterAttach(msg: Message): void {
  //   console.log('attached');
  // }

  public get contentHandler(): ContentHandler {
    return this._contentHandler;
  }

  constructor(contentHandler: ContentHandler) {
    super();
    this._contentHandler = contentHandler;
    this.id = 'jp-Greenflow-NodeEditor';
    this.addClass('jp-Greenflow-first');
  }

  protected render(): React.ReactElement<any> {
    const init: IEditorProp = {
      edges: [],
      nodeDatum: {},
      setChartState: null,
      nodes: [],
      handler: null
    };
    return (
      <React.Fragment>
        <UseSignal
          signal={this._contentHandler.updateEditor}
          initialArgs={init}
        >
          {(_, args): React.ReactElement<any> => {
            return (
              <NodeEditor
                handler={args.handler}
                nodeDatum={args.nodeDatum}
                setChartState={args.setChartState}
                nodes={args.nodes}
                edges={args.edges}
              />
            );
          }}
        </UseSignal>
      </React.Fragment>
    );
  }
}
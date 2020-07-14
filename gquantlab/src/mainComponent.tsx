// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget, UseSignal } from '@jupyterlab/apputils';
import React from 'react';
import { ContentHandler } from './document';
// eslint-disable-next-line @typescript-eslint/no-unused-vars

export class MainView extends ReactWidget {
  private _contentHandler: ContentHandler;

  constructor(contentHandler: ContentHandler) {
    super();
    this._contentHandler = contentHandler;
  }

  protected render(): React.ReactElement<any> {
    return (
      <div>
        {' '}
        this is it
        <UseSignal
          signal={this._contentHandler.contentChanged}
          initialArgs={{ allNodes: [], nodes: [], edges: [] }}
        >
          {(_, args): JSX.Element => {
            return (
              //<ChartEngine
              //  allNondes={args.allNodes}
              //  nodes={args.allNodes}
              //  edges={args.edges}
              ///>
              <div>
                <p>{JSON.stringify(args.allNodes)}</p>
              </div>
            );
          }}
        </UseSignal>
      </div>
    );
  }
}

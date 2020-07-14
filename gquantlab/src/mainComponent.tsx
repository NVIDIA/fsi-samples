// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget, UseSignal } from '@jupyterlab/apputils';
import React from 'react';
import { ContentHandler } from './document';
import { Widget } from '@lumino/widgets';
import { ChartEngine } from './chartEngine';

export class MainView extends ReactWidget {
  private _contentHandler: ContentHandler;
  private _height: number;
  private _width: number;

  constructor(contentHandler: ContentHandler) {
    super();
    this._contentHandler = contentHandler;
  }

  protected onResize(msg: Widget.ResizeMessage): void {
    this._height = msg.height;
    this._width = msg.width;
    this.render();
    this._contentHandler.emit();
  }

  protected render(): React.ReactElement<any> {
    console.log('re render');
    return (
      <div>
        <UseSignal
          signal={this._contentHandler.contentChanged}
          initialArgs={{ allNodes: {}, nodes: [], edges: [] }}
        >
          {(_, args): JSX.Element => {
            return (
              <ChartEngine
                contentHandler={this._contentHandler}
                height={this._height}
                width={this._width}
              />
            );
          }}
        </UseSignal>
      </div>
    );
  }
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget, UseSignal } from '@jupyterlab/apputils';
import React from 'react';
import { ContentHandler } from './document';
import { Widget } from '@lumino/widgets';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ChartEngine } from './chartEngine';
import { Message } from '@lumino/messaging';

export class MainView extends ReactWidget {
  private _contentHandler: ContentHandler;
  private _height: number;
  private _width: number;

  onAfterAttach(msg: Message): void {
    super.onAfterAttach(msg);
    this._height = 300;
    this._width = this.node.clientWidth;
    console.log('h', this._height, 'w', this._width);
    this.render();
    if (!this._contentHandler) {
      return;
    }
    if (!this._contentHandler.privateCopy) {
      return;
    }
    if (this._contentHandler.privateCopy.nodes.length === 0) {
      return;
    }

    this._contentHandler.renderNodesAndEdges(this._contentHandler.privateCopy);
    this._contentHandler.reLayoutSignalInstance.emit();
  }

  public get contentHandler(): ContentHandler {
    return this._contentHandler;
  }

  constructor(contentHandler: ContentHandler) {
    super();
    this._contentHandler = contentHandler;
    this.addClass('jp-GQuant');
  }

  protected onResize(msg: Widget.ResizeMessage): void {
    this._height = msg.height;
    this._width = msg.width;
    if (this._height < 0 || this._width < 0) {
      return;
    }
    this.render();
    this._contentHandler.emit();
  }

  protected render(): React.ReactElement<any> {
    console.log('re render');
    return (
      <div>
        <UseSignal
          signal={this._contentHandler.contentChanged}
          initialArgs={{ nodes: [], edges: [], width: null, height: null }}
        >
          {(_, args): JSX.Element => {
            return (
              <ChartEngine
                contentHandler={this._contentHandler}
                height={args.height ? args.height : this._height}
                width={args.width ? args.width : this._width}
              />
            );
          }}
        </UseSignal>
      </div>
    );
  }
}

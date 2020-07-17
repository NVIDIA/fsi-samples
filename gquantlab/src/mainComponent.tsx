// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget, UseSignal } from '@jupyterlab/apputils';
import React from 'react';
import { ContentHandler } from './document';
import { Widget } from '@lumino/widgets';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ChartEngine } from './chartEngine';
import { Message } from '@lumino/messaging';

const OUTPUT_CELL_HEIGHT = 300;

export class MainView extends ReactWidget {
  private _contentHandler: ContentHandler;
  private _height: number;
  private _width: number;

  private _mimerenderWidgetUpdateSize(): void {
    this._height = OUTPUT_CELL_HEIGHT;
    this._width = this.node.clientWidth;
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

  onAfterAttach(msg: Message): void {
    super.onAfterAttach(msg);
    this._mimerenderWidgetUpdateSize();
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
      // this is a hack that onResize doesn't work for rendered widget
      this._mimerenderWidgetUpdateSize();
      return;
    }
    this._contentHandler.emit();
  }

  protected render(): React.ReactElement<any> {
    console.log('re render');
    return (
      <div>
        <UseSignal
          signal={this._contentHandler.contentChanged}
          initialArgs={{ nodes: [], edges: [], width: 100, height: 100 }}
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

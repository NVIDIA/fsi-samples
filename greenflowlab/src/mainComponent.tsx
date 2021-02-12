// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';
//import { ContentHandler, IChartInput } from './document';
import { ContentHandler } from './document';
import { Panel, Widget } from '@lumino/widgets';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { ChartEngine } from './chartEngine';
import { Message } from '@lumino/messaging';
import { Cell } from '@jupyterlab/cells';

const OUTPUT_CELL_HEIGHT = 300;

export const OUTPUT_COLLECTOR = 'collector_id_fd9567b6';
export const OUTPUT_TYPE = 'Output_Collector';

export class MainView extends ReactWidget {
  private _contentHandler: ContentHandler;
  private _height: number;
  private _width: number;
  firstTimeReSize: boolean;

  private _calclateSize(): void {
    this._height = OUTPUT_CELL_HEIGHT;
    const cell: Cell = this.parent.parent.parent.parent.parent.parent as Cell;
    this._width = cell.inputArea.editorWidget.node.clientWidth - 5;
    //this._width = this.parent.parent.parent.node.clientWidth;
    // console.log('nsize', this._height, this._width);
    if (!this._contentHandler) {
      return;
    }
    if (!this._contentHandler.privateCopy) {
      return;
    }
    if (this._contentHandler.aspectRatio) {
      this._height = this._width * this._contentHandler.aspectRatio;
    }
  }

  /**
   * This is used exclusively to handle the greenflowlab widget resize event in the cell output of the notebook
   *
   */
  public mimerenderWidgetUpdateSize(): void {
    this._calclateSize();
    if (!this._contentHandler.privateCopy) {
      return;
    }
    if (!this._contentHandler.privateCopy.get('cache')) {
      return;
    }
    if (!this._contentHandler.privateCopy.get('cache').nodes) {
      return;
    }
    this._contentHandler.sizeStateUpdate.emit({
      nodes: [],
      edges: [],
      width: this._width,
      height: this._height
    });
    // update the result size
    const pane = this.parent as Panel;
    if (pane.widgets.length === 3) {
      const view = pane.widgets[2];
      view.node.style.width = this._width + 'px';
    }

    // const content: IChartInput = this._contentHandler.privateCopy.get('cache');
    // content['width'] = this._width;
    // content['height'] = this._height;
    // this._contentHandler.contentChanged.emit(content);
  }

  onAfterAttach(msg: Message): void {
    console.log('attached');
    super.onAfterAttach(msg);
    this._calclateSize();
    if (!this._contentHandler.privateCopy) {
      return;
    }
    if (!this._contentHandler.privateCopy.get('value')) {
      return;
    }
    this._contentHandler.renderGraph(
      this._contentHandler.privateCopy.get('value'),
      this._width,
      this._height
    );
  }

  public get contentHandler(): ContentHandler {
    return this._contentHandler;
  }

  constructor(contentHandler: ContentHandler) {
    super();
    this._contentHandler = contentHandler;
    this.addClass('jp-Greenflow');
    this.firstTimeReSize = false;
  }

  protected onResize(msg: Widget.ResizeMessage): void {
    this._height = msg.height;
    this._width = msg.width;
    console.log('resize', this._height, this._width);
    if (this._height < 0 || this._width < 0) {
      // this is a hack that onResize doesn't work for rendered widget
      this.mimerenderWidgetUpdateSize();
      return;
    }
    if (!this.firstTimeReSize) {
      this._contentHandler.emit(this._width, this._height);
      this.firstTimeReSize = true;
    } else {
      this._contentHandler.sizeStateUpdate.emit({
        nodes: [],
        edges: [],
        width: this._width,
        height: this._height
      });
    }
  }

  protected render(): React.ReactElement<any> {
    console.log('re render');
    return (
      <div>
        <ChartEngine contentHandler={this._contentHandler} />
      </div>
    );
  }
}

import { ContentHandler } from './document';

import { StackedPanel, Widget } from '@lumino/widgets';
import { EditorView } from './editorWidget';

/**
 * The class name added to the panels.
 */
const PANEL_CLASS = 'jp-RovaPanel';

/**
 * A panel which has the ability to add other children.
 */
export class EditorPanel extends StackedPanel {
  constructor(handler: ContentHandler) {
    super();
    this._handler = handler;
    this.addClass(PANEL_CLASS);
    this.id = 'greenflow-editor-panel';
    this.title.label = 'Node Editor';
    this.title.closable = true;

    this._example = new EditorView(this._handler);

    this.addWidget(this._example);
  }

  protected onResize(msg: Widget.ResizeMessage): void {
    console.log('here');
    console.log(msg);
  }

  dispose(): void {
    super.dispose();
  }

  get handler(): ContentHandler {
    return this._handler;
  }

  private _handler: ContentHandler;
  private _example: EditorView;
}

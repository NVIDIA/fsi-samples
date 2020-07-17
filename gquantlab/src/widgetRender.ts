// Copyright (c) Jupyter Development Team.
// Distributed under the terms of the Modified BSD License.

import { IDisposable } from '@lumino/disposable';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { MainView } from './mainComponent';
import { ContentHandler } from './document';
import { Panel } from '@lumino/widgets';

/**
 * A renderer for widgets.
 */
export class GQuantWidgetRenderer extends Panel
  implements IRenderMime.IRenderer, IDisposable {
  constructor(options: IRenderMime.IRendererOptions) {
    super();
    this.mimeType = options.mimeType;
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const source: any = model.data[this.mimeType];

    // If there is no model id, the view was removed, so hide the node.
    if (source.model_id === '') {
      this.hide();
      return Promise.resolve();
    }
    const contentHandler = new ContentHandler(null);
    const widget = new MainView(contentHandler);
    const workflows = JSON.parse(source);
    widget.contentHandler.setPrivateCopy(workflows);
    this.addWidget(widget);
    // When the widget is disposed, hide this container and make sure we
    // change the output model to reflect the view was closed.
    widget.disposed.connect(() => {
      this.hide();
    });
  }

  /**
   * The mimetype being rendered.
   */
  readonly mimeType: string;
}

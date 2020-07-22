// Copyright (c) Yi Dong
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers
} from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';
import { ContentHandler, } from './document';
import { MainView } from './mainComponent';
import { Panel } from '@lumino/widgets';

export class GQuantModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: GQuantModel.model_name,
      _model_module: GQuantModel.model_module,
      _model_module_version: GQuantModel.model_module_version,
      _view_name: GQuantModel.view_name,
      _view_module: GQuantModel.view_module,
      _view_module_version: GQuantModel.view_module_version,
      value: [],
      cache: { nodes:[], edges: []}
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers
    // Add any extra serializers here
  };

  static model_name = 'GQuantModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'GQuantView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class GQuantView extends DOMWidgetView {
  private _contentHandler: ContentHandler;
  private _widget: MainView;

  render() {
    this._contentHandler = new ContentHandler(null);
    const pane = new Panel();
    this._widget = new MainView(this._contentHandler);
    pane.addWidget(this._widget);
    this.pWidget = pane;
    this._contentHandler.renderGraph(this.model.get('value'));
    this._contentHandler.setPrivateCopy(this.model);
    this.model.on('change:value', this.value_changed, this);
    this.model.on('change:cache', this.cache_changed, this);
  }

  protected getFigureSize(): DOMRect {
    const figureSize: DOMRect = this.el.getBoundingClientRect();
    return figureSize;
  }

  processPhosphorMessage(msg: any) {
    super.processPhosphorMessage.apply(this, msg);
    switch (msg.type) {
      case 'resize':
      case 'after-show':
      case 'after-attach':
        if (this.pWidget.isVisible) {
          console.log(this.getFigureSize());
        }
        break;
    }
  }

  value_changed() {
  }

  cache_changed() {
    this._contentHandler.chartStateUpdate.emit(this.model.get('cache'));
  }

}

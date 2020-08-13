/* eslint-disable @typescript-eslint/camelcase */
// Copyright (c) Yi Dong
// Distributed under the terms of the Modified BSD License.

import { DOMWidgetModel, DOMWidgetView, ViewList } from '@jupyter-widgets/base';
import * as widgets from '@jupyter-widgets/base';
import { MODULE_NAME, MODULE_VERSION } from './version';
import { ContentHandler } from './document';
import { MainView } from './mainComponent';
import { Panel, Widget } from '@lumino/widgets';
import { CommandRegistry } from '@lumino/commands';
import { Toolbar, CommandToolbarButton } from '@jupyterlab/apputils';

export class GQuantModel extends DOMWidgetModel {
  static serializers = {
    ...DOMWidgetModel.serializers,
    sub: { deserialize: widgets.unpack_models }
  };

  defaults(): any {
    return {
      ...super.defaults(),
      _model_name: GQuantModel.model_name,
      _model_module: GQuantModel.model_module,
      _model_module_version: GQuantModel.model_module_version,
      _view_name: GQuantModel.view_name,
      _view_module: GQuantModel.view_module,
      _view_module_version: GQuantModel.view_module_version,
      value: [],
      cache: { nodes: [], edges: [] },
      sub: null
    };
  }

  static model_name = 'GQuantModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'GQuantView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class GQuantView extends DOMWidgetView {
  static commands: CommandRegistry = null;
  private _contentHandler: ContentHandler;
  private _widget: MainView;
  views: ViewList<DOMWidgetView>;

  addCommands(toolBar: Toolbar, contentHandler: ContentHandler): void {
    const commands = GQuantView.commands;
    const LAYOUT_COMMAND = 'gquant:toolbarReLayout';
    const COMMANDEXECUTE = 'gquant:toolbarexecute';
    const COMMANDCLEAN = 'gquant:toolbarcleanResult';
    const OPENTASKGRAPH = 'gquant:toolbaropenTaskGraph';
    const CREATEFILE = 'gquant:toolbarConvertCellToFile';

    const item0 = new CommandToolbarButton({
      commands: commands,
      id: LAYOUT_COMMAND
    });
    const item1 = new CommandToolbarButton({
      commands: commands,
      id: COMMANDEXECUTE
    });
    const item2 = new CommandToolbarButton({
      commands: commands,
      id: COMMANDCLEAN
    });
    const item3 = new CommandToolbarButton({
      commands: commands,
      id: OPENTASKGRAPH
    });
    const item4 = new CommandToolbarButton({
      commands: commands,
      id: CREATEFILE
    });

    toolBar.addItem('relayout', item0);
    toolBar.addItem('run', item1);
    toolBar.addItem('clean', item2);
    toolBar.addItem('open', item3);
    toolBar.addItem('create', item4);
  }

  render(): void {
    this._contentHandler = new ContentHandler(null);
    if (GQuantView.commands) {
      this._contentHandler.commandRegistry = GQuantView.commands;
    }
    this._contentHandler.runGraph.connect(this.run, this);
    this._contentHandler.cleanResult.connect(this.clean, this);
    const pane = new Panel();
    this._widget = new MainView(this._contentHandler);
    const toolBar = new Toolbar<Widget>();
    pane.addWidget(this._widget);
    pane.addWidget(toolBar);
    this.pWidget = pane;
    this._contentHandler.renderGraph(this.model.get('value'));
    this._contentHandler.setPrivateCopy(this.model);
    this.model.on('change:cache', this.cache_changed, this);
    this.views = new ViewList<DOMWidgetView>(this.addView, null, this);
    this.model.on('change:sub', this.sub_changed, this);
    this.addCommands(toolBar, this._contentHandler);
    //this.views.update([this.model.get('sub')]);
  }

  run(): void {
    this.send({
      event: 'run'
    });
  }

  clean(): void {
    this.send({
      event: 'clean'
    });
  }

  sub_changed(model: DOMWidgetModel, value: any): void {
    const subView = this.create_child_view(value);
    subView.then(view => {
      const pane = this.pWidget as Panel;
      if (pane.widgets.length === 3) {
        pane.layout.removeWidget(pane.widgets[2]);
      }
      pane.insertWidget(2, view.pWidget);
    });
    this.views.update([value]);
  }

  protected getFigureSize(): DOMRect {
    const figureSize: DOMRect = this.el.getBoundingClientRect();
    return figureSize;
  }

  addView(model: DOMWidgetModel): Promise<DOMWidgetView> {
    return this.create_child_view(model);
  }

  processPhosphorMessage(msg: any): void {
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

  cache_changed(): void {
    this._contentHandler.chartStateUpdate.emit(this.model.get('cache'));
  }
}

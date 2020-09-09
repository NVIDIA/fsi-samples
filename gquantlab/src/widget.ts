/* eslint-disable @typescript-eslint/camelcase */

import { DOMWidgetModel, DOMWidgetView, ViewList } from '@jupyter-widgets/base';
import * as widgets from '@jupyter-widgets/base';
import { MODULE_NAME, MODULE_VERSION } from './version';
import { ContentHandler, INode } from './document';
import { MainView } from './mainComponent';
import { Panel, Widget } from '@lumino/widgets';
import { CommandRegistry } from '@lumino/commands';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { Toolbar, CommandToolbarButton } from '@jupyterlab/apputils';
import {
  COMMAND_TOOL_BAR_RELAYOUT,
  COMMAND_TOOL_BAR_EXECUTE,
  COMMAND_TOOL_BAR_CLEAN,
  COMMAND_TOOL_BAR_OPEN_NEW_FILE,
  COMMAND_TOOL_BAR_CONVERT_CELL_TO_FILE,
  COMMAND_TOOL_BAR_INCLUDE_NEW_FILE,
  setupToolBarCommands
} from './commands';
import { JupyterFrontEnd } from '@jupyterlab/application';

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
  static apps: JupyterFrontEnd = null;
  static browserFactory: IFileBrowserFactory = null;
  private _contentHandler: ContentHandler;
  private _widget: MainView;
  views: ViewList<DOMWidgetView>;

  addCommands(commands: CommandRegistry, toolBar: Toolbar): void {
    const item0 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_RELAYOUT
    });
    const item1 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_EXECUTE
    });
    const item2 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_CLEAN
    });
    const item3 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_OPEN_NEW_FILE
    });
    const item4 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_CONVERT_CELL_TO_FILE
    });
    const item5 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_INCLUDE_NEW_FILE
    });

    toolBar.addItem('relayout', item0);
    toolBar.addItem('run', item1);
    toolBar.addItem('clean', item2);
    toolBar.addItem('open', item3);
    toolBar.addItem('add', item5);
    toolBar.addItem('create', item4);
  }

  render(): void {
    this._contentHandler = new ContentHandler(null);

    const commands = new CommandRegistry();
    this._contentHandler.commandRegistry = commands; // use its local command registry
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
    setupToolBarCommands(
      commands,
      this._contentHandler,
      GQuantView.browserFactory,
      GQuantView.apps.commands,
      GQuantView.apps
    );
    this.addCommands(commands, toolBar);
  }

  run(): void {
    this.send({
      event: 'run'
    });
  }

  clean(): void {
    const cache = this.model.get('cache');
    cache.nodes.forEach((node: INode) => {
      node.busy = false;
    });
    this._contentHandler.chartStateUpdate.emit(cache);
    this._contentHandler.privateCopy.set('cache', cache);
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

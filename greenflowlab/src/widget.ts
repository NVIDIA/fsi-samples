/* eslint-disable @typescript-eslint/camelcase */

import { DOMWidgetModel, DOMWidgetView, ViewList } from '@jupyter-widgets/base';
import * as widgets from '@jupyter-widgets/base';
import { MODULE_NAME, MODULE_VERSION } from './version';
import { requestAPI } from './greenflowlab';
import { ContentHandler, INode } from './document';
import { MainView } from './mainComponent';
import { Panel, Widget } from '@lumino/widgets';
import { CommandRegistry } from '@lumino/commands';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';
// import { Message } from '@lumino/messaging';

import { Toolbar, CommandToolbarButton } from '@jupyterlab/apputils';
import {
  COMMAND_TOOL_BAR_RELAYOUT,
  COMMAND_TOOL_BAR_EXECUTE,
  COMMAND_TOOL_BAR_CLEAN,
  COMMAND_TOOL_BAR_OPEN_NEW_FILE,
  COMMAND_TOOL_BAR_CONVERT_CELL_TO_FILE,
  COMMAND_TOOL_BAR_INCLUDE_NEW_FILE,
  COMMAND_TOOL_BAR_SHOW_LOG,
  setupToolBarCommands
} from './commands';
import { JupyterFrontEnd } from '@jupyterlab/application';
import { ContextMenuSvg } from '@jupyterlab/ui-components';
import { setupContextMenu } from '.';
import { Cell } from '@jupyterlab/cells';

export class GreenflowModel extends DOMWidgetModel {
  static serializers = {
    ...DOMWidgetModel.serializers,
    sub: { deserialize: widgets.unpack_models }
  };

  defaults(): any {
    return {
      ...super.defaults(),
      _model_name: GreenflowModel.model_name,
      _model_module: GreenflowModel.model_module,
      _model_module_version: GreenflowModel.model_module_version,
      _view_name: GreenflowModel.view_name,
      _view_module: GreenflowModel.view_module,
      _view_module_version: GreenflowModel.view_module_version,
      value: [],
      cache: { nodes: [], edges: [] },
      sub: null
    };
  }

  static model_name = 'GreenflowModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'GreenflowView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class GreenflowView extends DOMWidgetView {
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
    const item6 = new CommandToolbarButton({
      commands: commands,
      id: COMMAND_TOOL_BAR_SHOW_LOG
    });

    toolBar.addItem('relayout', item0);
    toolBar.addItem('run', item1);
    toolBar.addItem('clean', item2);
    toolBar.addItem('open', item3);
    toolBar.addItem('add', item5);
    toolBar.addItem('create', item4);
    toolBar.addItem('log', item6);
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
      GreenflowView.browserFactory,
      GreenflowView.apps.commands,
      GreenflowView.apps
    );
    this.addCommands(commands, toolBar);
    this.createContexMenu();
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
      // const message: Message = new Message('resize');
      //this._widget.processMessage(message);
    });
    this.views.update([value]).then(() => {
      const pane = this.pWidget as Panel;
      if (pane.widgets.length === 3) {
        const w = pane.widgets[2];
        const cell: Cell = this._widget.parent.parent.parent.parent.parent
          .parent as Cell;
        const _width = cell.inputArea.editorWidget.node.clientWidth - 5;
        w.node.style.width = _width + 'px';
      }
    });
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
    const cache = this.model.get('cache');
    if ('register' in cache) {
      const payload = cache['register'];
      const result = requestAPI<any>('register_node', {
        body: JSON.stringify(payload),
        method: 'POST'
      });
      result.then(data => {
        console.log(data);
      });
      delete cache['register'];
    }

    this._contentHandler.chartStateUpdate.emit(cache);
  }

  createContexMenu(): void {
    const commands = GreenflowView.apps.commands;
    const contextMenu = new ContextMenuSvg({ commands });
    setupContextMenu(contextMenu, commands, null);
    const createOuputView = 'notebook:create-output-view';

    contextMenu.addItem({
      type: 'separator',
      selector: '.jp-Greenflow'
    });

    contextMenu.addItem({
      command: createOuputView,
      selector: '.jp-Greenflow'
    });

    this.pWidget.node.addEventListener('contextmenu', (event: MouseEvent) => {
      if (contextMenu.open(event)) {
        event.preventDefault();
        event.stopPropagation();
      }
    });
  }
}

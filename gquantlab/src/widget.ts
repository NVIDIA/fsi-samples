// Copyright (c) Yi Dong
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers
} from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';
import { ContentHandler, IAllNodes } from './document';
import { MainView } from './mainComponent';
import { Panel, ContextMenu, Menu } from '@lumino/widgets';
import { CommandRegistry } from '@lumino/commands';
import { requestAPI } from './gquantlab';

// Import the CSS
// import '../css/widget.css';

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
      value: []
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

  render() {
    this._contentHandler = new ContentHandler(null);
    const pane = new Panel();
    const widget = new MainView(this._contentHandler);
    pane.addWidget(widget);
    this.pWidget = pane;
    //    this.el.classList.add('custom-widget');
    this.value_changed();
    this._contentHandler.setPrivateCopy(this.model);
    this.model.on('change:value', this.value_changed, this);
    this.createContexMenu();
  }

  createContexMenu() {
    const commands = new CommandRegistry();
    const contextMenu = new ContextMenu({ commands });

    commands.addCommand('gquant:reLayout', {
      label: 'Taskgraph Nodes Auto Layout',
      caption: 'Taskgraph Nodes Auto Layout',
      mnemonic: 0,
      execute: () => {
        this._contentHandler.reLayoutSignal.emit();
      }
    });

    contextMenu.addItem({
      command: 'gquant:reLayout',
      selector: '.jp-GQuant'
    });

    contextMenu.addItem({
      type: 'separator',
      selector: '.jp-GQuant'
    });

    const allNodes = requestAPI<any>('all_nodes');
    allNodes.then((allNodes: IAllNodes) => {
      for (const k in allNodes) {
        const submenu = new Menu({ commands });
        submenu.title.label = k;
        submenu.title.mnemonic = 0;
        for (let i = 0; i < allNodes[k].length; i++) {
          const name = allNodes[k][i].type;
          const commandName = 'addnode:' + name;
          commands.addCommand(commandName, {
            label: 'Add ' + name,
            mnemonic: 1,
            execute: () => {
              this._contentHandler.nodeAddedSignal.emit(allNodes[k][i]);
            }
          });
          submenu.addItem({ command: commandName });
        }
        contextMenu.addItem({
          type: 'submenu',
          submenu: submenu,
          selector: '.jp-GQuant'
        });
      }
    });
    this.pWidget.node.addEventListener('contextmenu', (event: MouseEvent) => {
      if (contextMenu.open(event)) {
        event.preventDefault();
        event.stopPropagation();
      }
    });
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
    console.log(this.model.get('value'));
    this._contentHandler.renderGraph(this.model.get('value'));
  }
}
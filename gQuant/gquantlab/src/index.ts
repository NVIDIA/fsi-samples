import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { Token, ReadonlyJSONObject } from '@lumino/coreutils';

import { MODULE_NAME, MODULE_VERSION } from './version';
import * as widgetExports from './widget';

import { requestAPI } from './gquantlab';

import gqStr from '../style/gq.svg';
import runStr from '../style/run.svg';
import cleanStr from '../style/clean.svg';
import layoutStr from '../style/layout.svg';

import {
  LabIcon,
  downloadIcon,
  saveIcon,
  editIcon,
  notebookIcon,
  addIcon,
  ContextMenuSvg
} from '@jupyterlab/ui-components';

import { CommandRegistry } from '@lumino/commands';

import { ToolbarButton } from '@jupyterlab/apputils';

import { toArray } from '@lumino/algorithm';
import {
  ICommandPalette,
  IWidgetTracker,
  WidgetTracker
} from '@jupyterlab/apputils';
import { GquantWidget, GquantFactory, IAllNodes } from './document';
import { Menu, Widget } from '@lumino/widgets';
import { IJupyterWidgetRegistry } from '@jupyter-widgets/base';
import { INotebookTracker } from '@jupyterlab/notebook';
import { CodeCell } from '@jupyterlab/cells';
import { MainView } from './mainComponent';
import '../style/editor.css';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import {
  COMMAND_CONVERT_CELL_TO_FILE,
  setupCommands,
  COMMAND_OPEN_NEW_FILE,
  COMMAND_INCLUDE_NEW_FILE,
  COMMAND_NEW_TASK_GRAPH,
  COMMAND_NEW_OPEN_TASK_GRAPH,
  COMMAND_OPEN_EDITOR,
  COMMAND_OPEN_NEW_NOTEBOOK,
  COMMAND_ADD_NODE,
  COMMAND_RELAYOUT,
  COMMAND_CHANGE_ASPECT_RATIO,
  COMMAND_EXECUTE,
  COMMAND_CLEAN,
  COMMAND_ADD_OUTPUT_COLLECTOR
} from './commands';
import { registerValidator } from './validator';
import { registerDisplay } from './showType';

// import { LabIcon } from '@jupyterlab/ui-components';
// import { LabIcon } from '@jupyterlab/ui-components/lib/icon/labicon';

//const WIDGET_VIEW_MIMETYPE = 'application/gquant-taskgraph';

export const FACTORY = 'GQUANTLAB';

type IGQUANTTracker = IWidgetTracker<GquantWidget>;

export const gqIcon = new LabIcon({ name: 'gquantlab:gq', svgstr: gqStr });

export const runIcon = new LabIcon({ name: 'gquantlab:gqrun', svgstr: runStr });

export const cleanIcon = new LabIcon({
  name: 'gquantlab:gqclean',
  svgstr: cleanStr
});

export const layoutIcon = new LabIcon({
  name: 'gquantlab:layout',
  svgstr: layoutStr
});

export const IGQUANTTracker = new Token<IGQUANTTracker>('gquant/tracki');

function outputEnabled(outputArea: any): boolean {
  if (!outputArea) {
    return false;
  }
  if (outputArea.widgets.length === 0) {
    return false;
  }
  let widget = outputArea.widgets[0];
  if (!widget) {
    return false;
  }

  if (widget.widgets[0] instanceof MainView) {
    return true;
  }

  const children = widget.children();
  if (!children) {
    return false;
  }
  // first one is output promot
  children.next();
  // second one is output wrapper
  widget = children.next();
  if (!widget) {
    return false;
  }
  // this is the panel
  widget = widget.children().next();
  if (!widget) {
    return false;
  }
  // this is the mainview
  const mainView = widget.children().next();
  if (!mainView) {
    return false;
  }
  if (!(mainView instanceof MainView)) {
    return false;
  }
  return true;
}

export function isLinkedView(currentWidget: any): boolean {
  if (currentWidget) {
    for (const view of toArray(currentWidget.children())) {
      if (view instanceof Widget) {
        if (view.id.startsWith('LinkedOutputView-')) {
          for (const outputs of toArray(view.children())) {
            for (const output of toArray(outputs.children())) {
              if (outputEnabled(output)) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

export function isEnabled(cell: any): boolean {
  if (!cell) {
    return false;
  }
  if (!(cell instanceof CodeCell)) {
    return false;
  }
  const codecell = cell as CodeCell;
  const outputArea = codecell.outputArea;
  return outputEnabled(outputArea);
}

export async function setupContextMenu(
  contextMenu: ContextMenuSvg,
  commands: CommandRegistry,
  palette: ICommandPalette
): Promise<Menu> {
  contextMenu.addItem({
    command: COMMAND_OPEN_EDITOR,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: 'docmanager:save',
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: 'filebrowser:download',
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_CONVERT_CELL_TO_FILE,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_OPEN_NEW_FILE,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_INCLUDE_NEW_FILE,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_RELAYOUT,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  const addNodeMenu = new Menu({ commands });
  addNodeMenu.title.label = 'Add Nodes';
  addNodeMenu.title.mnemonic = 4;
  const subMenuDict: { [key: string]: Menu } = {};
  const allNodes: IAllNodes = await requestAPI<any>('all_nodes');
  //allNodes.then((allNodes: IAllNodes) => {
    for (const k in allNodes) {
      const splits = k.split('.');
      let subMenu: Menu = null;
      for (let i = 0; i < splits.length; i++) {
        const key = splits.slice(0, i + 1).join('.');
        if (key in subMenuDict) {
          subMenu = subMenuDict[key];
        } else {
          subMenu = new Menu({ commands });
          subMenu.title.label = splits[i];
          subMenu.title.mnemonic = 0;
          subMenuDict[key] = subMenu;
          if (i > 0) {
            // add this submenu to parent
            const parentKey = splits.slice(0, i).join('.');
            const pMenu = subMenuDict[parentKey];
            pMenu.addItem({
              type: 'submenu',
              submenu: subMenu
            });
          } else {
            addNodeMenu.addItem({
              type: 'submenu',
              submenu: subMenu
            });
          }
        }
      }

      //const submenu = new Menu({ commands });
      //submenu.title.label = k;
      //submenu.title.mnemonic = 0;
      for (let i = 0; i < allNodes[k].length; i++) {
        const name = allNodes[k][i].type;
        const args = {
          name: name,
          node: (allNodes[k][i] as unknown) as ReadonlyJSONObject
        };
        subMenu.addItem({
          command: COMMAND_ADD_NODE,
          args: args
        });
        if (palette) {
          palette.addItem({
            command: COMMAND_ADD_NODE,
            category: 'Add New Nodes',
            args: args
          });
        }
      }
      // addNodeMenu.addItem({
      //   type: 'submenu',
      //   submenu: submenu
      // });
    }
  //});

  contextMenu.addItem({
    type: 'submenu',
    submenu: addNodeMenu,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_EXECUTE,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_CLEAN,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    command: COMMAND_ADD_OUTPUT_COLLECTOR,
    selector: '.jp-GQuant'
  });

  contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  const submenu = new Menu({ commands });
  submenu.title.label = 'Change Aspect Ratio';
  submenu.title.mnemonic = 0;

  [0.3, 0.5, 0.7, 1.0].forEach(d => {
    submenu.addItem({
      command: COMMAND_CHANGE_ASPECT_RATIO,
      args: { aspect: d }
    });
  });

  contextMenu.addItem({
    type: 'submenu',
    submenu: submenu,
    selector: '.jp-GQuant'
  });

  const plugins = requestAPI<any>('register_plugins');
  plugins.then( (d: any) =>{
    console.log(d['validation']);
    for (const k in d['validation']){
      let fun = new Function("required", "outputs", d['validation'][k]);
      registerValidator(k, fun);
    }

    for (const k in d['display']){
      let fun = new Function("metaObj", d['display'][k]);
      registerDisplay(k, fun);
    }
  });

  return addNodeMenu;
}

/**
 * Initialization data for the gquantlab extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'gquantlab-extension',
  requires: [
    IFileBrowserFactory,
    ILayoutRestorer,
    IMainMenu,
    ICommandPalette,
    INotebookTracker
  ],
  optional: [ILauncher],
  autoStart: true,
  activate: activateFun
};

const gquantWidget: JupyterFrontEndPlugin<void> = {
  id: 'gquantlab',
  requires: [IFileBrowserFactory, IJupyterWidgetRegistry],
  optional: [ILauncher],
  autoStart: true,
  activate: activateWidget
};

async function activateFun(
  app: JupyterFrontEnd,
  browserFactory: IFileBrowserFactory,
  restorer: ILayoutRestorer,
  menu: IMainMenu,
  palette: ICommandPalette,
  notebookTracker: INotebookTracker,
  launcher: ILauncher | null
): Promise<void> {
  const namespace = 'gquant';
  const factory = new GquantFactory({
    name: FACTORY,
    fileTypes: ['gq.yaml'],
    defaultFor: ['gq.yaml'],
    toolbarFactory: (widget: Widget): DocumentRegistry.IToolbarItem[] => {
      const main = widget as MainView;
      main.contentHandler.commandRegistry = app.commands;

      const layoutCallback = (): void => {
        const mainView = widget as MainView;
        mainView.contentHandler.reLayoutSignal.emit();
      };

      const downloadCallback = (): void => {
        app.commands.execute('filebrowser:download');
        // const mainView = widget as MainView;
        // mainView.contentHandler.reLayoutSignal.emit();
      };

      const saveCallback = (): void => {
        app.commands.execute('docmanager:save');
      };

      const notebookCallback = (): void => {
        app.commands.execute(COMMAND_OPEN_NEW_NOTEBOOK);
      };

      const editorCallback = (): void => {
        app.commands.execute(COMMAND_OPEN_EDITOR);
      };

      const addGraphCallback = (): void => {
        app.commands.execute(COMMAND_INCLUDE_NEW_FILE);
      };

      const layout = new ToolbarButton({
        className: 'myButton',
        icon: layoutIcon,
        onClick: layoutCallback,
        tooltip: 'Taskgraph Nodes Auto Layout'
      });

      const download = new ToolbarButton({
        className: 'myButton',
        icon: downloadIcon,
        onClick: downloadCallback,
        tooltip: 'Download the TaskGraph'
      });

      const save = new ToolbarButton({
        className: 'myButton',
        icon: saveIcon,
        onClick: saveCallback,
        tooltip: 'Save the TaskGraph'
      });

      const notebook = new ToolbarButton({
        className: 'myButton',
        icon: notebookIcon,
        onClick: notebookCallback,
        tooltip: 'Convert TaskGraph to Notebook'
      });

      const editor = new ToolbarButton({
        className: 'myButton',
        icon: editIcon,
        onClick: editorCallback,
        tooltip: 'Open Task Node Editor'
      });

      const addGraph = new ToolbarButton({
        className: 'myButton',
        icon: addIcon,
        onClick: addGraphCallback,
        tooltip: 'Import a TaskGraph from file'
      });

      return [
        { name: 'layout', widget: layout },
        { name: 'download', widget: download },
        { name: 'save', widget: save },
        { name: 'notebook', widget: notebook },
        { name: 'editor', widget: editor },
        { name: 'add', widget: addGraph }
      ];
    }
  });
  const { commands } = app;
  const tracker = new WidgetTracker<GquantWidget>({ namespace });

  function getMainView(): MainView {
    const currentWidget = app.shell.currentWidget;
    if (currentWidget) {
      for (const view of toArray(currentWidget.children())) {
        if (view.id.startsWith('LinkedOutputView-')) {
          for (const outputs of toArray(view.children())) {
            for (const output of toArray(outputs.children())) {
              const out = output as any;
              if (
                out &&
                out.widgets.length > 0 &&
                out.widgets[0].widgets.length > 0 &&
                out.widgets[0].widgets[0] instanceof MainView
              ) {
                const mainView = out.widgets[0].widgets[0] as MainView;
                return mainView;
              }
            }
          }
        }
      }
    }

    const codecell = notebookTracker.activeCell as CodeCell;
    const outputArea = codecell.outputArea;
    let widget = outputArea.widgets[0];
    const children = widget.children();
    //first one is output promot
    children.next();
    //second one is output wrapper
    widget = children.next();
    // this is the panel
    widget = widget.children().next();
    // this is the mainview
    const mainView = widget.children().next() as MainView;
    return mainView;
  }

  /**
   * Whether there is an active graph editor
   */
  function isCellVisible(): boolean {
    const currentWidget = app.shell.currentWidget;
    if (isLinkedView(currentWidget)) {
      return true;
    }
    const cellVisible =
      notebookTracker.currentWidget !== null &&
      notebookTracker.currentWidget === app.shell.currentWidget;
    return cellVisible;
  }

  /**
   * Whether there is an active graph editor
   */
  function isGquantVisible(): boolean {
    const gquantVisible =
      tracker.currentWidget !== null &&
      tracker.currentWidget === app.shell.currentWidget;
    return gquantVisible;
  }

  // Handle state restoration.
  restorer.restore(tracker, {
    command: 'docmanager:open',
    args: widget => ({ path: widget.context.path, factory: FACTORY }),
    name: widget => widget.context.path
  });

  factory.widgetCreated.connect((sender, widget) => {
    widget.title.icon = gqIcon;
    // Notify the instance tracker if restore data needs to update.
    widget.context.pathChanged.connect(() => {
      tracker.save(widget);
    });
    tracker.add(widget);
  });
  app.docRegistry.addWidgetFactory(factory);

  // register the filetype
  app.docRegistry.addFileType({
    name: 'gq.yaml',
    displayName: 'TaskGraph',
    mimeTypes: ['application/gq.yaml'],
    extensions: ['.gq.yaml'],
    //iconClass: 'jp-MaterialIcon jp-ImageIcon',
    icon: gqIcon,
    fileFormat: 'text'
  });

  setupCommands(
    commands,
    app,
    getMainView,
    browserFactory,
    isCellVisible,
    isGquantVisible,
    notebookTracker
  );


  // Add a launcher item if the launcher is available.
  if (launcher) {
    launcher.add({
      command: COMMAND_NEW_OPEN_TASK_GRAPH,
      rank: 1,
      category: 'Other'
    });
  }

  const addNodeMenu = setupContextMenu(app.contextMenu, commands, palette);

  if (menu) {
    // Add new text file creation to the file menu.
    menu.fileMenu.newMenu.addGroup(
      [{ command: COMMAND_NEW_OPEN_TASK_GRAPH }],
      40
    );
    //palette.addItem({ command: 'gquant:export-yaml', category: 'Notebook Operations', args: args });
    menu.fileMenu.addGroup([{ command: COMMAND_NEW_TASK_GRAPH }], 40);
    menu.addMenu(await addNodeMenu, { rank: 40 });
  }

  if (palette) {
    const args = { format: 'YAML', label: 'YAML', isPalette: true };
    palette.addItem({
      command: COMMAND_NEW_TASK_GRAPH,
      category: 'Notebook Operations',
      args: args
    });
    palette.addItem({
      command: COMMAND_RELAYOUT,
      category: 'GquantLab',
      args: args
    });
    [0.3, 0.5, 0.7, 1.0].forEach(d => {
      palette.addItem({
        command: COMMAND_CHANGE_ASPECT_RATIO,
        category: 'GquantLab',
        args: { aspect: d }
      });
    });
    palette.addItem({
      command: COMMAND_ADD_OUTPUT_COLLECTOR,
      category: 'Add New Nodes',
      args: args
    });
    palette.addItem({
      command: COMMAND_CONVERT_CELL_TO_FILE,
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: COMMAND_CLEAN,
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: COMMAND_EXECUTE,
      category: 'GquantLab',
      args: args
    });
  }
  //add key board shortcuts
  app.commands.addKeyBinding({
    command: COMMAND_RELAYOUT,
    keys: ['Alt A'],
    selector: '.jp-GQuant'
  });
  app.commands.addKeyBinding({
    command: COMMAND_ADD_OUTPUT_COLLECTOR,
    keys: ['Alt O'],
    selector: '.jp-GQuant'
  });
  //add key board shortcuts
  app.commands.addKeyBinding({
    command: COMMAND_EXECUTE,
    keys: ['Alt R'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: COMMAND_CLEAN,
    keys: ['Alt C'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: COMMAND_RELAYOUT,
    keys: ['Alt A'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: COMMAND_ADD_OUTPUT_COLLECTOR,
    keys: ['Alt O'],
    selector: '.jp-Notebook'
  });

  //add key board shortcuts
  app.commands.addKeyBinding({
    command: COMMAND_EXECUTE,
    keys: ['Alt R'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: COMMAND_CLEAN,
    keys: ['Alt C'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: COMMAND_RELAYOUT,
    keys: ['Alt A'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: COMMAND_ADD_OUTPUT_COLLECTOR,
    keys: ['Alt O'],
    selector: '.jp-LinkedOutputView'
  });

  app.contextMenu.addItem({
    command: COMMAND_OPEN_NEW_NOTEBOOK,
    selector: '.jp-GQuant'
  });
}

function activateWidget(
  app: JupyterFrontEnd,
  browserFactory: IFileBrowserFactory,
  jupyterWidgetRegistry: IJupyterWidgetRegistry
): void {
  // passing the commands registry
  widgetExports.GQuantView.apps = app;
  widgetExports.GQuantView.browserFactory = browserFactory;
  jupyterWidgetRegistry.registerWidget({
    name: MODULE_NAME,
    version: MODULE_VERSION,
    exports: widgetExports
  });
}

/**
 * Export the plugins as default.
 */
const plugins: JupyterFrontEndPlugin<any>[] = [extension, gquantWidget];
export default plugins;

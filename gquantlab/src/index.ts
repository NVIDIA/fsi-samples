import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { Token } from '@lumino/coreutils';

import { MODULE_NAME, MODULE_VERSION } from './version';

import * as widgetExports from './widget';

import { requestAPI } from './gquantlab';

import gqStr from '../style/gq.svg';

import { LabIcon } from '@jupyterlab/ui-components';


import {
  ICommandPalette,
  IWidgetTracker,
  WidgetTracker
} from '@jupyterlab/apputils';
import { GquantWidget, GquantFactory, IAllNodes } from './document';
import { Menu } from '@lumino/widgets';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IJupyterWidgetRegistry } from '@jupyter-widgets/base';
// import { LabIcon } from '@jupyterlab/ui-components';
// import { LabIcon } from '@jupyterlab/ui-components/lib/icon/labicon';

//const WIDGET_VIEW_MIMETYPE = 'application/gquant-taskgraph';

const FACTORY = 'GQUANTLAB';

type IGQUANTTracker = IWidgetTracker<GquantWidget>;

export const gqIcon = new LabIcon({ name: 'gquantlab:gq', svgstr: gqStr });

export const IGQUANTTracker = new Token<IGQUANTTracker>('gquant/tracki');
/**
 * Initialization data for the gquantlab extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'gquantlab',
  requires: [
    IFileBrowserFactory,
    ILayoutRestorer,
    IMainMenu,
    ICommandPalette,
    IRenderMimeRegistry,
    IJupyterWidgetRegistry
  ],
  optional: [ILauncher],
  autoStart: true,
  activate: activateFun
};

function activateFun(
  app: JupyterFrontEnd,
  browserFactory: IFileBrowserFactory,
  restorer: ILayoutRestorer,
  menu: IMainMenu,
  palette: ICommandPalette,
  rendermime: IRenderMimeRegistry,
  jupyterWidgetRegistry: IJupyterWidgetRegistry,
  launcher: ILauncher | null
): void {
  const namespace = 'gquant';
  const factory = new GquantFactory({
    name: FACTORY,
    fileTypes: ['gq.yaml'],
    defaultFor: ['gq.yaml']
  });
  const { commands } = app;
  const tracker = new WidgetTracker<GquantWidget>({ namespace });

  /**
   * Whether there is an active graph editor
   */
  function isEnabled(): boolean {
    return (
      tracker.currentWidget !== null &&
      tracker.currentWidget === app.shell.currentWidget
    );
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

  // Function to create a new untitled diagram file, given
  // the current working directory.
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const createGQFile = (cwd: string) => {
    return commands
      .execute('docmanager:new-untitled', {
        path: cwd,
        type: 'file',
        ext: '.gq.yaml'
      })
      .then(model => {
        return commands.execute('docmanager:open', {
          path: model.path,
          factory: FACTORY
        });
      });
  };

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const createNewTaskgraph = (cwd: string) => {
    return commands
      .execute('docmanager:new-untitled', {
        path: cwd,
        type: 'file',
        ext: '.gq.yaml'
      })
      .then(model => {
        //let wdg = app.shell.currentWidget as any;
        // wdg.getSVG()
        model.content = '';
        model.format = 'text';
        app.serviceManager.contents.save(model.path, model);
      });
  };

  // Add a command for creating a new diagram file.
  commands.addCommand('gquant:create-new', {
    label: 'TaskGraph',
    icon: gqIcon,
    caption: 'Create a new task graph file',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createGQFile(cwd);
    }
  });

  commands.addCommand('gquant:export-yaml', {
    label: 'Create an empty Taskgraph',
    caption: 'Create an empty Taskgraph',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createNewTaskgraph(cwd);
    },
    isEnabled
  });

  commands.addCommand('gquant:reLayout', {
    label: 'Taskgraph Nodes Auto Layout',
    caption: 'Taskgraph Nodes Auto Layout',
    mnemonic: 0,
    execute: () => {
      const wdg = app.shell.currentWidget as any;
      wdg.contentHandler.reLayoutSignal.emit();
    },
    isEnabled
  });

  app.contextMenu.addItem({
    command: 'docmanager:save',
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    command: 'filebrowser:download',
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    command: 'gquant:reLayout',
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
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
            const wdg = app.shell.currentWidget as any;
            wdg.contentHandler.nodeAddedSignal.emit(allNodes[k][i]);
          },
          isEnabled
        });
        submenu.addItem({ command: commandName });
        if (palette) {
          const args = { format: 'YAML', label: 'YAML', isPalette: true };
          palette.addItem({
            command: commandName,
            category: 'Add New Nodes',
            args: args
          });
        }
      }
      app.contextMenu.addItem({
        type: 'submenu',
        submenu: submenu,
        selector: '.jp-GQuant'
      });
    }
  });
  // Add a launcher item if the launcher is available.
  if (launcher) {
    launcher.add({
      command: 'gquant:create-new',
      rank: 1,
      category: 'Other'
    });
  }

  if (menu) {
    // Add new text file creation to the file menu.
    menu.fileMenu.newMenu.addGroup([{ command: 'gquant:create-new' }], 40);
    //palette.addItem({ command: 'gquant:export-yaml', category: 'Notebook Operations', args: args });
    menu.fileMenu.addGroup([{ command: 'gquant:export-yaml' }], 40);
  }

  if (palette) {
    const args = { format: 'YAML', label: 'YAML', isPalette: true };
    palette.addItem({
      command: 'gquant:export-yaml',
      category: 'Notebook Operations',
      args: args
    });
    palette.addItem({
      command: 'gquant:reLayout',
      category: 'Notebook Operations',
      args: args
    });
  }

  jupyterWidgetRegistry.registerWidget({
    name: MODULE_NAME,
    version: MODULE_VERSION,
    exports: widgetExports
  });
}

// (app: JupyterFrontEnd, ) => {
//     console.log('JupyterLab extension gquantlab is activated!');
//
//     requestAPI<any>('get_example')
//       .then(data => {
//         console.log(data);
//       })
//       .catch(reason => {
//         console.error(
//           `The gquantlab server extension appears to be missing.\n${reason}`
//         );
//       });
//   }

export default extension;

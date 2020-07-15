import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { Token } from '@lumino/coreutils';
// import { requestAPI } from './gquantlab';
import {
  ICommandPalette,
  IWidgetTracker,
  WidgetTracker
} from '@jupyterlab/apputils';
import { GquantWidget, GquantFactory } from './document';

const FACTORY = 'GQUANTLAB';

type IGQUANTTracker = IWidgetTracker<GquantWidget>;

export const IGQUANTTracker = new Token<IGQUANTTracker>('gquant/tracki');
/**
 * Initialization data for the gquantlab extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'gquantlab',
  requires: [IFileBrowserFactory, ILayoutRestorer, IMainMenu, ICommandPalette],
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
   * Whether there is an active DrawIO editor.
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
    widget.title.icon = 'jp-MaterialIcon jp-ImageIcon'; // TODO change
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
    iconClass: 'jp-MaterialIcon jp-ImageIcon',
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
    iconClass: 'jp-MaterialIcon jp-ImageIcon',
    caption: 'Create a new task graph file',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createGQFile(cwd);
    }
  });

  commands.addCommand('gquant:export-yaml', {
    label: 'Export diagram as SVG',
    caption: 'Export diagram as SVG',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createNewTaskgraph(cwd);
    },
    isEnabled
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
  }
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

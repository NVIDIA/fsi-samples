import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { IRenderMimeRegistry } from '@jupyterlab/rendermime';

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
import { GquantWidget, GquantFactory, IAllNodes, INode } from './document';
import { Menu } from '@lumino/widgets';
import { IJupyterWidgetRegistry } from '@jupyter-widgets/base';
import { INotebookTracker } from '@jupyterlab/notebook';
import { CodeCell } from '@jupyterlab/cells';
import { MainView, OUTPUT_COLLECTOR } from './mainComponent';
import YAML from 'yaml';
import { OutputPanel } from './outputPanel';

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
  id: 'gquantlab-extension',
  requires: [
    IFileBrowserFactory,
    ILayoutRestorer,
    IMainMenu,
    ICommandPalette,
    INotebookTracker,
    IRenderMimeRegistry
  ],
  optional: [ILauncher],
  autoStart: true,
  activate: activateFun
};

const gquantWidget: JupyterFrontEndPlugin<void> = {
  id: 'gquantlab',
  requires: [IJupyterWidgetRegistry],
  optional: [ILauncher],
  autoStart: true,
  activate: activateWidget
};

function activateFun(
  app: JupyterFrontEnd,
  browserFactory: IFileBrowserFactory,
  restorer: ILayoutRestorer,
  menu: IMainMenu,
  palette: ICommandPalette,
  notebookTracker: INotebookTracker,
  rendermime: IRenderMimeRegistry,
  launcher: ILauncher | null
): void {
  const namespace = 'gquant';
  const factory = new GquantFactory({
    name: FACTORY,
    fileTypes: ['gq.yaml'],
    defaultFor: ['gq.yaml']
  });
  const { commands, shell } = app;
  const tracker = new WidgetTracker<GquantWidget>({ namespace });

  // function getRendermime(): IRenderMimeRegistry {
  //   const codecell = notebookTracker.activeCell as CodeCell;
  //   const outputArea = codecell.outputArea;
  //   return outputArea.rendermime;
  // }

  function getMainView(): MainView {
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
    return (
      notebookTracker.currentWidget !== null &&
      notebookTracker.currentWidget === app.shell.currentWidget
    );
  }

  /**
   * Whether there is an active graph editor
   */
  function isGquantVisible(): boolean {
    return (
      tracker.currentWidget !== null &&
      tracker.currentWidget === app.shell.currentWidget
    );
  }

  /**
   * Whether there is an active graph editor
   */
  function isVisible(): boolean {
    return isGquantVisible() || isCellVisible();
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
  const createGQFile = async (cwd: string) => {
    const model = await commands.execute('docmanager:new-untitled', {
      path: cwd,
      type: 'file',
      ext: '.gq.yaml'
    });
    return commands.execute('docmanager:open', {
      path: model.path,
      factory: FACTORY
    });
  };

  const convertToGQFile = async (cwd: string) => {
    const model = await commands.execute('docmanager:new-untitled', {
      path: cwd,
      type: 'file',
      ext: '.gq.yaml'
    });
    const mainView = getMainView();
    const obj = mainView.contentHandler.privateCopy.get('value');
    model.content = YAML.stringify(obj);
    model.format = 'text';
    app.serviceManager.contents.save(model.path, model);
  };

  commands.addCommand('gquant:convertCellToFile', {
    label: 'Create Taskgraph from this Cell',
    caption: 'Create Taskgraph from this Cell',
    icon: gqIcon,
    execute: () => {
      //const cwd = notebookTracker.currentWidget.context.path;
      const cwd = browserFactory.defaultBrowser.model.path;
      return convertToGQFile(cwd);
    },
    isVisible: isCellVisible
  });

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const createNewTaskgraph = async (cwd: string) => {
    const model = await commands.execute('docmanager:new-untitled', {
      path: cwd,
      type: 'file',
      ext: '.gq.yaml'
    });
    //let wdg = app.shell.currentWidget as any;
    // wdg.getSVG()
    model.content = '';
    model.format = 'text';
    app.serviceManager.contents.save(model.path, model);
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
    }
  });

  commands.addCommand('gquant:reLayout', {
    label: 'Taskgraph Nodes Auto Layout',
    caption: 'Taskgraph Nodes Auto Layout',
    mnemonic: 0,
    execute: () => {
      if (isCellVisible()) {
        const mainView = getMainView();
        mainView.contentHandler.reLayoutSignal.emit();
      } else {
        const wdg = app.shell.currentWidget as any;
        wdg.contentHandler.reLayoutSignal.emit();
      }
    },
    isVisible
  });

  commands.addCommand('gquant:aspect1', {
    label: 'AspectRatio 1.0',
    caption: 'AspectRatio 1.0',
    mnemonic: 14,
    execute: () => {
      const mainView = getMainView();
      mainView.contentHandler.aspectRatio = 1.0;
      mainView.mimerenderWidgetUpdateSize();
    },
    isVisible: isCellVisible
  });

  commands.addCommand('gquant:aspect0.3', {
    label: 'AspectRatio 0.3',
    caption: 'AspectRatio 0.3',
    mnemonic: 14,
    execute: () => {
      const mainView = getMainView();
      mainView.contentHandler.aspectRatio = 0.3;
      mainView.mimerenderWidgetUpdateSize();
    },
    isVisible: isCellVisible
  });

  commands.addCommand('gquant:aspect0.5', {
    label: 'AspectRatio 0.5',
    caption: 'AspectRatio 0.5',
    mnemonic: 14,
    execute: () => {
      const mainView = getMainView();
      mainView.contentHandler.aspectRatio = 0.5;
      mainView.mimerenderWidgetUpdateSize();
    },
    isVisible: isCellVisible
  });

  commands.addCommand('gquant:aspect0.7', {
    label: 'AspectRatio 0.7',
    caption: 'AspectRatio 0.7',
    mnemonic: 14,
    execute: () => {
      const mainView = getMainView();
      mainView.contentHandler.aspectRatio = 0.7;
      mainView.mimerenderWidgetUpdateSize();
    },
    isVisible: isCellVisible
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
    command: 'gquant:convertCellToFile',
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
          mnemonic: 4,
          execute: () => {
            if (isCellVisible()) {
              const mainView = getMainView();
              mainView.contentHandler.nodeAddedSignal.emit(allNodes[k][i]);
            } else {
              const wdg = app.shell.currentWidget as any;
              wdg.contentHandler.nodeAddedSignal.emit(allNodes[k][i]);
            }
          },
          isVisible
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

  app.contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  commands.addCommand('add:outputCollector', {
    label: 'Add Output Collector',
    mnemonic: 1,
    execute: () => {
      const output: INode = {
        id: OUTPUT_COLLECTOR,
        width: 160,
        type: 'Output Collector',
        conf: {},
        required: {},
        output_columns: [],
        outputs: [],
        schema: {},
        ui: {},
        inputs: [{ name: 'in1', type: ['any'] }]
      };
      if (isCellVisible()) {
        const mainView = getMainView();
        mainView.contentHandler.nodeAddedSignal.emit(output);
      }
      if (isGquantVisible()) {
        const wdg = app.shell.currentWidget as any;
        wdg.contentHandler.nodeAddedSignal.emit(output);
      }
    },
    isVisible: isVisible
  });

  app.contextMenu.addItem({
    command: 'add:outputCollector',
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  const submenu = new Menu({ commands });
  submenu.title.label = 'Change Aspect Ratio';
  submenu.title.mnemonic = 0;
  submenu.addItem({ command: 'gquant:aspect0.3' });
  submenu.addItem({ command: 'gquant:aspect0.5' });
  submenu.addItem({ command: 'gquant:aspect0.7' });
  submenu.addItem({ command: 'gquant:aspect1' });
  app.contextMenu.addItem({
    type: 'submenu',
    submenu: submenu,
    selector: '.jp-GQuant'
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
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: 'gquant:aspect0.3',
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: 'gquant:aspect0.5',
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: 'gquant:aspect0.7',
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: 'gquant:aspect1',
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: 'add:outputCollector',
      category: 'Add New Nodes',
      args: args
    });
    palette.addItem({
      command: 'gquant:convertCellToFile',
      category: 'GquantLab',
      args: args
    });
  }

  let panel: OutputPanel;
  const manager = app.serviceManager;

  /**
   * Creates a example panel.
   *
   * @returns The panel
   */
  async function createPanel(): Promise<OutputPanel> {
    panel = new OutputPanel(manager, rendermime);
    shell.add(panel, 'main');
    return panel;
  }

  const commandExecute = 'gquant:execute';
  const commandCreate = 'gquant:createOutput';
  // add commands to registry
  commands.addCommand(commandCreate, {
    label: 'Open the Output Panel',
    caption: 'Open the Output Panel',
    execute: createPanel
  });

  commands.addCommand(commandExecute, {
    label: 'Run',
    caption: 'Run',
    execute: async () => {
      // Create the panel if it does not exist
      if (!panel) {
        await createPanel();
      }
      let mainView: MainView;
      let objStr: string = '';
      if (isCellVisible()) {
        // Prompt the user about the statement to be executed
        mainView = getMainView();
        objStr = JSON.stringify(
          mainView.contentHandler.privateCopy.get('value')
        );
      }
      if (isGquantVisible()) {
        mainView = app.shell.currentWidget as any;
        objStr = JSON.stringify(
          YAML.parse(mainView.contentHandler.context.model.toString())
        );
      }
      const outputStr = JSON.stringify(mainView.contentHandler.outputs);
      console.log(outputStr);
      const input = `import json\nfrom gquant.dataframe_flow import TaskGraph\nobj="""${objStr}"""\ntaskList=json.loads(obj)\ntaskGraph=TaskGraph(taskList)\noutlist=${outputStr}\ntaskGraph.run(outlist, formated=True)`;
      // Execute the statement
      const code = input;
      panel.execute(code);
    }
  });

  app.contextMenu.addItem({
    command: commandExecute,
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    command: commandCreate,
    selector: '.jp-GQuant'
  });
}

function activateWidget(
  app: JupyterFrontEnd,
  jupyterWidgetRegistry: IJupyterWidgetRegistry
): void {
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

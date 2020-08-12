import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin,
  ILayoutRestorer
} from '@jupyterlab/application';

import { IFileBrowserFactory, FileDialog } from '@jupyterlab/filebrowser';

import { ILauncher } from '@jupyterlab/launcher';

import { IMainMenu } from '@jupyterlab/mainmenu';

import { Token } from '@lumino/coreutils';

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
  notebookIcon
} from '@jupyterlab/ui-components';

import { folderIcon } from '@jupyterlab/ui-components';

import { ToolbarButton } from '@jupyterlab/apputils';

import { IDisposable, DisposableDelegate } from '@lumino/disposable';
import { toArray } from '@lumino/algorithm';
import {
  ICommandPalette,
  IWidgetTracker,
  WidgetTracker
} from '@jupyterlab/apputils';
import {
  GquantWidget,
  GquantFactory,
  IAllNodes,
  INode,
  IChartInput
} from './document';
import { Menu, Widget } from '@lumino/widgets';
import { IJupyterWidgetRegistry } from '@jupyter-widgets/base';
import {
  INotebookTracker,
  NotebookPanel,
  INotebookModel
} from '@jupyterlab/notebook';
import { CodeCell } from '@jupyterlab/cells';
import { MainView, OUTPUT_COLLECTOR, OUTPUT_TYPE } from './mainComponent';
import YAML from 'yaml';
import { EditorPanel } from './EditorPanel';
import '../style/editor.css';
import { DocumentRegistry } from '@jupyterlab/docregistry';

// import { LabIcon } from '@jupyterlab/ui-components';
// import { LabIcon } from '@jupyterlab/ui-components/lib/icon/labicon';

//const WIDGET_VIEW_MIMETYPE = 'application/gquant-taskgraph';

const FACTORY = 'GQUANTLAB';

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

function isLinkedView(currentWidget: any): boolean {
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

function isEnabled(cell: any): boolean {
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
  launcher: ILauncher | null
): void {
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
        app.commands.execute('gquant:openAnNewNotebook');
      };

      const editorCallback = (): void => {
        app.commands.execute('gquant:openeditor');
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

      return [
        { name: 'layout', widget: layout },
        { name: 'download', widget: download },
        { name: 'save', widget: save },
        { name: 'notebook', widget: notebook },
        { name: 'editor', widget: editor }
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

  const convertToGQFile = async (cwd: string): Promise<void> => {
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

  commands.addCommand('gquant:selectTheFile', {
    label: 'Select the file',
    caption: 'Select the file',
    icon: folderIcon,
    execute: async (args: any) => {
      let dialog = null;
      if (args && 'filter' in args) {
        dialog = FileDialog.getOpenFiles({
          manager: browserFactory.defaultBrowser.model.manager, // IDocumentManager
          title: 'Select the File',
          filter: model =>
            args['filter'].some((d: string): boolean => model.path.endsWith(d))
        });
      } else {
        dialog = FileDialog.getOpenFiles({
          manager: browserFactory.defaultBrowser.model.manager, // IDocumentManager
          title: 'Select the File'
        });
      }
      const result = await dialog;
      if (result.button.accept) {
        // console.log(result.value);
        const values = result.value;
        if (values.length === 1) {
          return values[0];
        }
        //let files = result.value;
      }
    }
  });

  commands.addCommand('gquant:selectThePath', {
    label: 'Select the Path',
    caption: 'Select the Path',
    icon: folderIcon,
    execute: async () => {
      const dialog = FileDialog.getExistingDirectory({
        manager: browserFactory.defaultBrowser.model.manager, // IDocumentManager
        title: 'Select the Path'
      });
      const result = await dialog;
      if (result.button.accept) {
        // console.log(result.value);
        const values = result.value;
        if (values.length === 1) {
          return values[0];
        }
        //let files = result.value;
      }
    }
  });

  commands.addCommand('gquant:openNewFile', {
    label: 'Open TaskGraph file',
    caption: 'Open TaskGraph file',
    icon: folderIcon,
    execute: async () => {
      const dialog = FileDialog.getOpenFiles({
        manager: browserFactory.defaultBrowser.model.manager, // IDocumentManager
        filter: model => model.path.endsWith('.gq.yaml')
      });
      const result = await dialog;
      if (result.button.accept) {
        // console.log(result.value);
        const values = result.value;
        if (values.length === 1) {
          // only 1 file is allowed
          const payload = { path: values[0].path };
          const workflows: IChartInput = await requestAPI<any>(
            'load_graph_path',
            {
              body: JSON.stringify(payload),
              method: 'POST'
            }
          );
          const mainView = getMainView();
          mainView.contentHandler.contentReset.emit(workflows);
        }
        //let files = result.value;
      }
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

  function uuidv4(): string {
    return Math.random()
      .toString(36)
      .substring(2, 15);
  }

  app.commands.addCommand('gquant:openeditor', {
    label: 'Task Node Editor',
    caption: 'Open the Task Node Editor',
    icon: editIcon,
    mnemonic: 0,
    execute: () => {
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          let panel = null;
          for (const view of toArray<Widget>(app.shell.widgets())) {
            if (
              view instanceof EditorPanel &&
              (view as EditorPanel).handler === mainView.contentHandler
            ) {
              // console.log('found it');
              panel = view;
              break;
            }
          }
          if (panel === null) {
            panel = new EditorPanel(mainView.contentHandler);
            panel.id = panel.id + uuidv4();
            app.shell.add(panel, 'main', { mode: 'split-bottom' });
          } else {
            app.shell.activateById(panel.id);
          }
        } else if (isLinkedView(app.shell.currentWidget)) {
          const mainView = getMainView();
          let panel = null;
          for (const view of toArray<Widget>(app.shell.widgets())) {
            if (
              view instanceof EditorPanel &&
              (view as EditorPanel).handler === mainView.contentHandler
            ) {
              // console.log('found it');
              panel = view;
              break;
            }
          }
          if (panel === null) {
            panel = new EditorPanel(mainView.contentHandler);
            panel.id = panel.id + uuidv4();
            app.shell.add(panel, 'main', { mode: 'split-bottom' });
          } else {
            app.shell.activateById(panel.id);
          }
        }
      } else {
        const wdg = app.shell.currentWidget as any;
        wdg.contentHandler.reLayoutSignal.emit();
        let panel = null;
        for (const view of toArray<Widget>(app.shell.widgets())) {
          if (
            view instanceof EditorPanel &&
            (view as EditorPanel).handler === wdg.contentHandler
          ) {
            // console.log('found it');
            panel = view;
            break;
          }
        }
        if (panel === null) {
          panel = new EditorPanel(wdg.contentHandler);
          panel.id = panel.id + uuidv4();
          app.shell.add(panel, 'main', { mode: 'split-bottom' });
        } else {
          app.shell.activateById(panel.id);
        }
      }
    },
    isVisible
  });

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const createNewNotebook = async (input1: string, input2: string) => {
    const model = await commands.execute('docmanager:new-untitled', {
      type: 'notebook'
    });
    const empty: any[] = [];
    const notebook = {
      cells: [
        {
          cell_type: "code",
          execution_count: 1,
          metadata: {},
          outputs: empty,
          source: input1
        },
        {
          cell_type: "code",
          execution_count: 2,
          metadata: {},
          outputs: empty,
          source: input2
        }
      ],
      metadata: {
        kernelspec: {
          display_name: 'Python 3',
          language: 'python',
          name: 'python3'
        }
      },
      nbformat: 4,
      nbformat_minor: 4
    };

    model.content = notebook;
    model.format = 'text';
    const savedModel = await app.serviceManager.contents.save(
      model.path,
      model
    );
    browserFactory.defaultBrowser.model.manager.open(savedModel.path);
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
    icon: layoutIcon,
    mnemonic: 0,
    execute: () => {
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          mainView.contentHandler.reLayoutSignal.emit();
        } else if (isLinkedView(app.shell.currentWidget)) {
          const mainView = getMainView();
          mainView.contentHandler.reLayoutSignal.emit();
        }
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
    isEnabled: isCellVisible
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
    isEnabled: isCellVisible
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
    isEnabled: isCellVisible
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
    isEnabled: isCellVisible
  });

  app.contextMenu.addItem({
    command: 'gquant:openeditor',
    selector: '.jp-GQuant'
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
    command: 'gquant:openNewFile',
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

  const addNodeMenu = new Menu({ commands });
  addNodeMenu.title.label = 'Add Nodes';
  addNodeMenu.title.mnemonic = 4;
  const allNodes = requestAPI<any>('all_nodes');
  allNodes.then((allNodes: IAllNodes) => {
    for (const k in allNodes) {
      const submenu = new Menu({ commands });
      submenu.title.label = k;
      submenu.title.mnemonic = 0;
      for (let i = 0; i < allNodes[k].length; i++) {
        const name = allNodes[k][i].type;
        const commandName = `addnode:${k}.${name}`;
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
      addNodeMenu.addItem({
        type: 'submenu',
        submenu: submenu
      });
    }
  });

  app.contextMenu.addItem({
    type: 'submenu',
    submenu: addNodeMenu,
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    type: 'separator',
    selector: '.jp-GQuant'
  });

  const commandExecute = 'gquant:execute';
  const commandClean = 'gquant:cleanResult';

  commands.addCommand(commandExecute, {
    label: 'Run',
    caption: 'Run',
    icon: runIcon,
    mnemonic: 0,
    execute: async () => {
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          mainView.contentHandler.saveCache.emit();
          mainView.contentHandler.runGraph.emit();
        } else if (isLinkedView(app.shell.currentWidget)) {
          const mainView = getMainView();
          mainView.contentHandler.saveCache.emit();
          mainView.contentHandler.runGraph.emit();
        }
      }
    },
    isVisible: isCellVisible
  });

  commands.addCommand(commandClean, {
    label: 'Clean Result',
    caption: 'Clean Result',
    icon: cleanIcon,
    mnemonic: 0,
    execute: async () => {
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          mainView.contentHandler.cleanResult.emit();
        }
      }
    },
    isVisible: isCellVisible
  });

  app.contextMenu.addItem({
    command: commandExecute,
    selector: '.jp-GQuant'
  });

  app.contextMenu.addItem({
    command: commandClean,
    selector: '.jp-GQuant'
  });

  commands.addCommand('add:outputCollector', {
    label: 'Add Output Collector',
    mnemonic: 1,
    execute: () => {
      const output: INode = {
        id: OUTPUT_COLLECTOR,
        width: 160,
        type: OUTPUT_TYPE,
        conf: {},
        required: {},
        // eslint-disable-next-line @typescript-eslint/camelcase
        output_columns: [],
        outputs: [],
        schema: {},
        ui: {},
        inputs: [{ name: 'in1', type: ['any'] }]
      };
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          mainView.contentHandler.nodeAddedSignal.emit(output);
        }
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
    menu.addMenu(addNodeMenu, { rank: 40 });
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
    palette.addItem({
      command: commandClean,
      category: 'GquantLab',
      args: args
    });
    palette.addItem({
      command: commandExecute,
      category: 'GquantLab',
      args: args
    });
  }
  //add key board shortcuts
  app.commands.addKeyBinding({
    command: 'gquant:reLayout',
    keys: ['Alt A'],
    selector: '.jp-GQuant'
  });
  app.commands.addKeyBinding({
    command: 'add:outputCollector',
    keys: ['Alt O'],
    selector: '.jp-GQuant'
  });
  //add key board shortcuts
  app.commands.addKeyBinding({
    command: commandExecute,
    keys: ['Alt R'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: commandClean,
    keys: ['Alt C'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: 'gquant:reLayout',
    keys: ['Alt A'],
    selector: '.jp-Notebook'
  });
  app.commands.addKeyBinding({
    command: 'add:outputCollector',
    keys: ['Alt O'],
    selector: '.jp-Notebook'
  });

  //add key board shortcuts
  app.commands.addKeyBinding({
    command: commandExecute,
    keys: ['Alt R'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: commandClean,
    keys: ['Alt C'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: 'gquant:reLayout',
    keys: ['Alt A'],
    selector: '.jp-LinkedOutputView'
  });
  app.commands.addKeyBinding({
    command: 'add:outputCollector',
    keys: ['Alt O'],
    selector: '.jp-LinkedOutputView'
  });

  commands.addCommand('gquant:openAnNewNotebook', {
    label: 'Convert TaskGraph to Notebook',
    caption: 'Convert TaskGraph to Notebook',
    icon: notebookIcon,
    execute: async () => {
      let mainView: MainView;
      let objStr = '';
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
          YAML.parse(mainView.contentHandler.context.model.toString()),
          null,
          2
        );
      }
      const input1 = `import json\nfrom gquant.dataframe_flow import TaskGraph\nobj="""${objStr}"""\ntaskList=json.loads(obj)\ntaskGraph=TaskGraph(taskList)\ntaskGraph.draw()`;
      const input2 = 'taskGraph.run(formated=True)';
      return createNewNotebook(input1, input2);
      // Execute the statement
    },
    isVisible: isGquantVisible
  });
  app.contextMenu.addItem({
    command: 'gquant:openAnNewNotebook',
    selector: '.jp-GQuant'
  });
}

function activateWidget(
  app: JupyterFrontEnd,
  jupyterWidgetRegistry: IJupyterWidgetRegistry
): void {
  // passing the commands registry
  widgetExports.GQuantView.commands = app.commands;
  jupyterWidgetRegistry.registerWidget({
    name: MODULE_NAME,
    version: MODULE_VERSION,
    exports: widgetExports
  });
}

/**
 * A notebook widget extension that adds a button to the toolbar.
 */
export class ButtonExtension
  implements DocumentRegistry.IWidgetExtension<NotebookPanel, INotebookModel> {
  /**
   * Create a new extension object.
   */
  createNew(
    panel: NotebookPanel,
    _context: DocumentRegistry.IContext<INotebookModel>
  ): IDisposable {
    function getMainView(): MainView {
      const codecell = panel.content.activeCell as CodeCell;
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

    const callback = (): void => {
      if (isEnabled(panel.content.activeCell)) {
        const mainView = getMainView();
        mainView.contentHandler.saveCache.emit();
        mainView.contentHandler.runGraph.emit();
      }
    };

    const layoutCallback = (): void => {
      if (isEnabled(panel.content.activeCell)) {
        const mainView = getMainView();
        mainView.contentHandler.saveCache.emit();
        mainView.contentHandler.reLayoutSignal.emit();
      }
    };

    const button = new ToolbarButton({
      className: 'myButton',
      icon: runIcon,
      onClick: callback,
      tooltip: 'Run GQuant TaskGraph'
    });

    const button2 = new ToolbarButton({
      className: 'myButton',
      icon: layoutIcon,
      onClick: layoutCallback,
      tooltip: 'Taskgraph Nodes Auto Layout'
    });

    panel.toolbar.insertItem(0, 'runAll', button);
    panel.toolbar.insertAfter('runAll', 'layout', button2);

    return new DisposableDelegate(() => {
      button.dispose();
      button2.dispose();
    });
  }
}

/**
 * Activate the extension.
 */
function activate(app: JupyterFrontEnd): void {
  app.docRegistry.addWidgetExtension('Notebook', new ButtonExtension());
}

/**
 * The plugin registration information.
 */
const buttonExtension: JupyterFrontEndPlugin<void> = {
  activate,
  id: 'my-extension-name:buttonPlugin',
  autoStart: true
};

/**
 * Export the plugins as default.
 */
const plugins: JupyterFrontEndPlugin<any>[] = [
  extension,
  gquantWidget,
  buttonExtension
];
export default plugins;

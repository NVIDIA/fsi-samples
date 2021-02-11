/* eslint-disable @typescript-eslint/camelcase */
import { JupyterFrontEnd } from '@jupyterlab/application';
//import YAML from 'yaml';
import jsyaml from 'js-yaml';
import { CommandRegistry } from '@lumino/commands';
import {
  gqIcon,
  FACTORY,
  isLinkedView,
  isEnabled,
  layoutIcon,
  cleanIcon,
  runIcon
} from '.';
import { listIcon } from '@jupyterlab/ui-components';
import { IFileBrowserFactory, FileDialog } from '@jupyterlab/filebrowser';
import {
  folderIcon,
  editIcon,
  notebookIcon,
  addIcon
} from '@jupyterlab/ui-components';
import { IChartInput, INode, ContentHandler } from './document';
import { requestAPI } from './greenflowlab';
import { Widget } from '@lumino/widgets';
import { EditorPanel } from './EditorPanel';
import { toArray } from '@lumino/algorithm';
import { INotebookTracker } from '@jupyterlab/notebook';
import { MainView, OUTPUT_COLLECTOR, OUTPUT_TYPE } from './mainComponent';

export const COMMAND_CONVERT_CELL_TO_FILE = 'greenflow:convertCellToFile';
export const COMMAND_TOOL_BAR_CONVERT_CELL_TO_FILE =
  'greenflow:toolbarConvertCellToFile';
export const COMMAND_SELECT_FILE = 'greenflow:selectTheFile';
export const COMMAND_SELECT_PATH = 'greenflow:selectThePath';
export const COMMAND_OPEN_NEW_FILE = 'greenflow:openNewFile';
export const COMMAND_INCLUDE_NEW_FILE = 'greenflow:includeNewFile';
export const COMMAND_TOOL_BAR_INCLUDE_NEW_FILE = 'greenflow:includeNewFile';
export const COMMAND_TOOL_BAR_OPEN_NEW_FILE = 'greenflow:toolbaropenTaskGraph';
export const COMMAND_NEW_TASK_GRAPH = 'greenflow:export-yaml';
export const COMMAND_NEW_OPEN_TASK_GRAPH = 'greenflow:create-new';
export const COMMAND_OPEN_EDITOR = 'greenflow:openeditor';
export const COMMAND_OPEN_NEW_NOTEBOOK = 'greenflow:openAnNewNotebook';
export const COMMAND_ADD_NODE = 'greenflow:addnode';
export const COMMAND_RELAYOUT = 'greenflow:reLayout';
export const COMMAND_TOOL_BAR_RELAYOUT = 'greenflow:toolbarReLayout';
export const COMMAND_CHANGE_ASPECT_RATIO = 'greenflow:aspect';
export const COMMAND_EXECUTE = 'greenflow:execute';
export const COMMAND_CLEAN = 'greenflow:cleanResult';
export const COMMAND_TOOL_BAR_EXECUTE = 'greenflow:toolbarexecute';
export const COMMAND_TOOL_BAR_CLEAN = 'greenflow:toolbarcleanResult';
export const COMMAND_ADD_OUTPUT_COLLECTOR = 'greenflow:addOutputCollector';
export const COMMAND_OPEN_DOC_FORWARD = 'greenflow:openDocumentForward';
export const COMMAND_CREATE_CUST_NODE = 'greenflow:createCustomizedNode';
export const COMMAND_TOOL_BAR_SHOW_LOG = 'greenflow:showLogBox';

function uuidv4(): string {
  return Math.random()
    .toString(36)
    .substring(2, 15);
}

export function setupCommands(
  commands: CommandRegistry,
  app: JupyterFrontEnd,
  getMainView: Function,
  browserFactory: IFileBrowserFactory,
  isCellVisible: any,
  isGreenflowVisible: any,
  notebookTracker: INotebookTracker
): void {
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  const createNewNotebook = async (
    input1: string,
    input2: string,
    input3: string
  ) => {
    const model = await commands.execute('docmanager:new-untitled', {
      type: 'notebook'
    });
    const empty: any[] = [];
    const notebook = {
      cells: [
        {
          cell_type: 'code',
          execution_count: 1,
          metadata: {},
          outputs: empty,
          source: input1
        },
        {
          cell_type: 'code',
          execution_count: 2,
          metadata: {},
          outputs: empty,
          source: input2
        },
        {
          cell_type: 'code',
          execution_count: 3,
          metadata: {},
          outputs: empty,
          source: input3
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

  /**
   * Whether there is an active graph editor
   */
  function isVisible(): boolean {
    return isGreenflowVisible() || isCellVisible();
  }

  const convertToGQFile = async (cwd: string): Promise<void> => {
    const model = await commands.execute('docmanager:new-untitled', {
      path: cwd,
      type: 'file',
      ext: '.gq.yaml'
    });
    const mainView = getMainView();
    const obj = mainView.contentHandler.privateCopy.get('value');
    model.content = jsyaml.safeDump(obj);
    model.format = 'text';
    app.serviceManager.contents.save(model.path, model);
  };

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

  const createNewTaskgraph = async (cwd: string): Promise<void> => {
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

  commands.addCommand(COMMAND_CONVERT_CELL_TO_FILE, {
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

  commands.addCommand(COMMAND_SELECT_FILE, {
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
      }
    }
  });

  commands.addCommand(COMMAND_SELECT_PATH, {
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
        const values = result.value;
        if (values.length === 1) {
          return values[0];
        }
      }
    }
  });

  commands.addCommand(COMMAND_CREATE_CUST_NODE, {
    label: 'Create Customized Node',
    caption: 'Create Customized Node',
    icon: folderIcon,
    execute: async args => {
      const cwd = browserFactory.defaultBrowser.model.path;
      const model = await commands.execute('docmanager:new-untitled', {
        path: cwd,
        type: 'file',
        ext: '.py'
      });
      const confContent = JSON.stringify({ conf: args['conf'] }, null, 2);
      const fileContent = `import greenflow
from greenflow.dataframe_flow.portsSpecSchema import ConfSchema
import json

data = """${confContent}
"""


class ${args['nodeName']}(greenflow.plugin_nodes.util.CompositeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modify the self.conf to the one that this Composite node wants
        global data
        node_conf = self.conf
        data_obj = json.loads(data)
        data_obj['conf']['subnodes_conf'].update(node_conf)
        self.conf = data_obj['conf']

    def conf_schema(self):
        full_schema = super().conf_schema()
        full_schema_json = full_schema.json
        ui = full_schema.ui
        json = {
            "title": "${args['nodeName']} configure",
            "type": "object",
            "description": "Enter your node description here",
            "properties": {
            }
        }
        item_dict = full_schema_json['properties']["subnodes_conf"]['properties']
        for key in item_dict.keys():
            json['properties'][key] = item_dict[key]
        return ConfSchema(json=json, ui=ui)

      `;
      model.content = fileContent;
      model.format = 'text';
      app.serviceManager.contents.save(model.path, model);
    }
  });

  commands.addCommand(COMMAND_OPEN_NEW_FILE, {
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
          const mainView = getMainView() as MainView;
          mainView.contentHandler.contentReset.emit(workflows);
        }
        //let files = result.value;
      }
    },
    isVisible: isCellVisible
  });

  commands.addCommand(COMMAND_INCLUDE_NEW_FILE, {
    label: 'Import TaskGraph from file',
    caption: 'Import TaskGraph from file',
    icon: addIcon,
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
          if (isCellVisible()) {
            const mainView = getMainView() as MainView;
            mainView.contentHandler.includeContent.emit(workflows);
          } else if (isGreenflowVisible()) {
            const mainView = app.shell.currentWidget as any;
            mainView.contentHandler.includeContent.emit(workflows);
          }
        }
        //let files = result.value;
      }
    }
  });

  commands.addCommand(COMMAND_NEW_TASK_GRAPH, {
    label: 'Create an empty Taskgraph',
    caption: 'Create an empty Taskgraph',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createNewTaskgraph(cwd);
    }
  });

  // Add a command for creating a new diagram file.
  commands.addCommand(COMMAND_NEW_OPEN_TASK_GRAPH, {
    label: 'TaskGraph',
    icon: gqIcon,
    caption: 'Create a new task graph file',
    execute: () => {
      const cwd = browserFactory.defaultBrowser.model.path;
      return createGQFile(cwd);
    }
  });

  app.commands.addCommand(COMMAND_OPEN_EDITOR, {
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

  commands.addCommand(COMMAND_OPEN_NEW_NOTEBOOK, {
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
      if (isGreenflowVisible()) {
        mainView = app.shell.currentWidget as any;
        objStr = JSON.stringify(
          jsyaml.safeLoad(mainView.contentHandler.context.model.toString()),
          null,
          2
        );
      }
      const input1 = `# run this cell if you need Dask
from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster()
from dask.distributed import Client
client = Client(cluster)
client
`;
      const input2 = `import json
from greenflow.dataframe_flow import TaskGraph
obj="""${objStr}"""
taskList=json.loads(obj)
taskGraph=TaskGraph(taskList)
taskGraph.draw()`;
      const input3 = 'taskGraph.run(formated=True)';
      return createNewNotebook(input1, input2, input3);
      // Execute the statement
    },
    isVisible: isGreenflowVisible
  });

  commands.addCommand(COMMAND_ADD_NODE, {
    label: args => 'Add ' + args.name,
    mnemonic: 4,
    execute: args => {
      if (isCellVisible()) {
        const mainView = getMainView();
        mainView.contentHandler.nodeAddedSignal.emit(args.node);
      } else {
        const wdg = app.shell.currentWidget as any;
        wdg.contentHandler.nodeAddedSignal.emit(args.node);
      }
    },
    isVisible
  });

  commands.addCommand(COMMAND_RELAYOUT, {
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

  commands.addCommand(COMMAND_CHANGE_ASPECT_RATIO, {
    label: args => 'AspectRatio ' + args.aspect.toString(),
    caption: 'AspectRatio 1.0',
    mnemonic: 14,
    execute: args => {
      const mainView = getMainView();
      mainView.contentHandler.aspectRatio = args.aspect;
      mainView.mimerenderWidgetUpdateSize();
    },
    isEnabled: isCellVisible
  });

  commands.addCommand(COMMAND_EXECUTE, {
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

  commands.addCommand(COMMAND_CLEAN, {
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

  commands.addCommand(COMMAND_ADD_OUTPUT_COLLECTOR, {
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
        output_meta: [],
        outputs: [],
        schema: {},
        ui: {},
        inputs: [{ name: 'in1', type: [['any']] }] // TODO, come back to use dynamic approach
      };
      if (isCellVisible()) {
        if (isEnabled(notebookTracker.activeCell)) {
          const mainView = getMainView();
          mainView.contentHandler.nodeAddedSignal.emit(output);
        }
      }
      if (isGreenflowVisible()) {
        const wdg = app.shell.currentWidget as any;
        wdg.contentHandler.nodeAddedSignal.emit(output);
      }
    },
    isVisible: isVisible
  });

  commands.addCommand(COMMAND_OPEN_DOC_FORWARD, {
    label: 'Task Node Editor',
    caption: 'Open the Task Node Editor',
    icon: editIcon,
    mnemonic: 0,
    execute: args => {
      app.commands.execute('docmanager:open', args);
    }
  });
}

export function setupToolBarCommands(
  commands: CommandRegistry,
  contentHandler: ContentHandler,
  browserFactory: IFileBrowserFactory,
  systemCommands: CommandRegistry,
  app: JupyterFrontEnd
): void {
  const convertToGQFile = async (cwd: string): Promise<void> => {
    const model = await systemCommands.execute('docmanager:new-untitled', {
      path: cwd,
      type: 'file',
      ext: '.gq.yaml'
    });
    const obj = contentHandler.privateCopy.get('value');
    model.content = jsyaml.safeDump(obj);
    model.format = 'text';
    app.serviceManager.contents.save(model.path, model);
  };

  commands.addCommand(COMMAND_TOOL_BAR_OPEN_NEW_FILE, {
    label: '',
    caption: 'Open TaskGraph file',
    icon: folderIcon,
    mnemonic: 0,
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
          contentHandler.contentReset.emit(workflows);
        }
        //let files = result.value;
      }
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_CONVERT_CELL_TO_FILE, {
    label: '',
    caption: 'Create Taskgraph from this Cell',
    icon: gqIcon,
    execute: async (args: any) => {
      //const cwd = notebookTracker.currentWidget.context.path;
      const cwd = browserFactory.defaultBrowser.model.path;
      return convertToGQFile(cwd);
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_RELAYOUT, {
    label: '',
    caption: 'Taskgraph Nodes Auto Layout',
    icon: layoutIcon,
    mnemonic: 0,
    execute: () => {
      contentHandler.reLayoutSignal.emit();
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_EXECUTE, {
    label: '',
    caption: 'Run',
    icon: runIcon,
    mnemonic: 0,
    execute: async () => {
      contentHandler.saveCache.emit();
      contentHandler.runGraph.emit();
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_SHOW_LOG, {
    label: '',
    caption: 'Show Log Console',
    icon: listIcon,
    mnemonic: 0,
    execute: async () => {
      app.commands.execute('logconsole:open', {
        insertMode: 'split-bottom',
        // eslint-disable-next-line prettier/prettier
        ref: app.shell.currentWidget?.id
      });
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_CLEAN, {
    label: '',
    caption: 'Clean Result',
    icon: cleanIcon,
    mnemonic: 0,
    execute: async () => {
      contentHandler.cleanResult.emit();
    }
  });

  commands.addCommand(COMMAND_SELECT_FILE, {
    label: 'Select the file',
    caption: 'Select the file',
    icon: folderIcon,
    execute: async (args: any) => {
      const value = await app.commands.execute(COMMAND_SELECT_FILE, args);
      return value;
    }
  });

  commands.addCommand(COMMAND_SELECT_PATH, {
    label: 'Select the Path',
    caption: 'Select the Path',
    icon: folderIcon,
    execute: async () => {
      const value = app.commands.execute(COMMAND_SELECT_PATH);
      return value;
    }
  });

  commands.addCommand(COMMAND_OPEN_EDITOR, {
    label: 'Task Node Editor',
    caption: 'Open the Task Node Editor',
    icon: editIcon,
    mnemonic: 0,
    execute: () => {
      if (isLinkedView(app.shell.currentWidget)) {
        let panel = null;
        for (const view of toArray<Widget>(app.shell.widgets())) {
          if (
            view instanceof EditorPanel &&
            (view as EditorPanel).handler === contentHandler
          ) {
            // console.log('found it');
            panel = view;
            break;
          }
        }
        if (panel === null) {
          panel = new EditorPanel(contentHandler);
          panel.id = panel.id + uuidv4();
          app.shell.add(panel, 'main', { mode: 'split-bottom' });
        } else {
          app.shell.activateById(panel.id);
        }
      } else {
        let panel = null;
        for (const view of toArray<Widget>(app.shell.widgets())) {
          if (
            view instanceof EditorPanel &&
            (view as EditorPanel).handler === contentHandler
          ) {
            // console.log('found it');
            panel = view;
            break;
          }
        }
        if (panel === null) {
          panel = new EditorPanel(contentHandler);
          panel.id = panel.id + uuidv4();
          app.shell.add(panel, 'main', { mode: 'split-bottom' });
        } else {
          app.shell.activateById(panel.id);
        }
      }
    }
  });
  commands.addCommand(COMMAND_OPEN_DOC_FORWARD, {
    label: 'Task Node Editor',
    caption: 'Open the Task Node Editor',
    icon: editIcon,
    mnemonic: 0,
    execute: args => {
      app.commands.execute('docmanager:open', args);
    }
  });
  commands.addCommand(COMMAND_CREATE_CUST_NODE, {
    label: 'Create Customized Node',
    caption: 'Create Customized Node',
    icon: folderIcon,
    execute: args => {
      app.commands.execute(COMMAND_CREATE_CUST_NODE, args);
    }
  });

  commands.addCommand(COMMAND_TOOL_BAR_INCLUDE_NEW_FILE, {
    label: '',
    caption: 'Import TaskGraph from file',
    icon: addIcon,
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
          contentHandler.includeContent.emit(workflows);
        }
        //let files = result.value;
      }
    }
  });
}

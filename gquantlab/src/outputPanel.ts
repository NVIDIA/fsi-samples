import {
  ISessionContext,
  SessionContext,
  sessionContextDialogs
} from '@jupyterlab/apputils';

import { OutputAreaModel, SimplifiedOutputArea } from '@jupyterlab/outputarea';

import { IRenderMimeRegistry } from '@jupyterlab/rendermime';

import { KernelMessage, ServiceManager } from '@jupyterlab/services';

import { Message } from '@lumino/messaging';

import { StackedPanel } from '@lumino/widgets';

// import { WidgetRenderer, WidgetManager } from '@jupyter-widgets/jupyterlab-manager'

/**
 * The class name added to the example panel.
 */
const PANEL_CLASS = 'jp-RovaPanel';

/**
 * A panel with the ability to add other children.
 */
export class OutputPanel extends StackedPanel {
  constructor(
    manager: ServiceManager.IManager,
    rendermime: IRenderMimeRegistry
  ) {
    super();
    this.addClass(PANEL_CLASS);
    this.id = 'kernel-output-panel';
    this.title.label = 'Gquant Output';
    this.title.closable = true;

    this._sessionContext = new SessionContext({
      sessionManager: manager.sessions,
      specsManager: manager.kernelspecs,
      name: 'Kernel Output'
    });

    this._outputareamodel = new OutputAreaModel();
    this._outputarea = new SimplifiedOutputArea({
      model: this._outputareamodel,
      rendermime: rendermime
    });

    this.addWidget(this._outputarea);

    void this._sessionContext
      .initialize()
      .then(async value => {
        if (value) {
          await sessionContextDialogs.selectKernel(this._sessionContext);
        }
      })
      .catch(reason => {
        console.error(
          `Failed to initialize the session in ExamplePanel.\n${reason}`
        );
      });
  }

  get session(): ISessionContext {
    return this._sessionContext;
  }

  dispose(): void {
    this._sessionContext.dispose();
    super.dispose();
  }

  execute(code: string): void {
    SimplifiedOutputArea.execute(code, this._outputarea, this._sessionContext)
      .then((msg: KernelMessage.IExecuteReplyMsg) => {
        console.log(msg);
      })
      .catch(reason => console.error(reason));
  }

  protected onCloseRequest(msg: Message): void {
    super.onCloseRequest(msg);
    this.dispose();
  }

  private _sessionContext: SessionContext;
  private _outputarea: SimplifiedOutputArea;
  private _outputareamodel: OutputAreaModel;
}

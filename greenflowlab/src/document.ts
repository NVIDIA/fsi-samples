import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './greenflowlab';
import jsyaml from 'js-yaml';
import { IEditorProp } from './nodeEditor';
import { Signal } from '@lumino/signaling';
import { MainAreaWidget } from '@jupyterlab/apputils';
import { WidgetModel } from '@jupyter-widgets/base';
import { CommandRegistry } from '@lumino/commands';

export class ContentHandler {
  private _outputs: string[]; // store a list of outputs
  context: DocumentRegistry.Context;
  // this one will relayout the graph
  private _contentChanged = new Signal<ContentHandler, IChartInput>(this);

  // this reset the content, leave the size as before
  private _contentReset = new Signal<ContentHandler, IChartInput>(this);

  // this add extra graph to current graph, leave the size as before
  private _includeContent = new Signal<ContentHandler, IChartInput>(this);

  // this one just update the graphs, no relayout
  private _chartStateUpdate = new Signal<ContentHandler, IChartInput>(this);
  // this one just update the size of the chart
  private _sizeStateUpdate = new Signal<ContentHandler, IChartInput>(this);

  private _runGraph = new Signal<ContentHandler, void>(this);
  private _cleanResult = new Signal<ContentHandler, void>(this);

  private _updateEditor = new Signal<ContentHandler, IEditorProp>(this);

  // signal used to sync the graph status to server
  private _saveCache = new Signal<ContentHandler, void>(this);

  // create a signal that emits the added node command
  private _nodeAdded = new Signal<ContentHandler, INode>(this);

  private _privateCopy: WidgetModel;

  // create a signal that emits the relayout command
  private _reLayout = new Signal<ContentHandler, void>(this);

  private _aspectRatio = 0.3;

  private _commandRegistry: CommandRegistry;

  set commandRegistry(commandRegistry: CommandRegistry) {
    this._commandRegistry = commandRegistry;
  }

  get commandRegistry(): CommandRegistry {
    return this._commandRegistry;
  }

  get updateEditor(): Signal<ContentHandler, IEditorProp> {
    return this._updateEditor;
  }

  get runGraph(): Signal<ContentHandler, void> {
    return this._runGraph;
  }

  get saveCache(): Signal<ContentHandler, void> {
    return this._saveCache;
  }

  get cleanResult(): Signal<ContentHandler, void> {
    return this._cleanResult;
  }

  get reLayoutSignal(): Signal<ContentHandler, void> {
    return this._reLayout;
  }

  set aspectRatio(aspectRatio: number) {
    this._aspectRatio = aspectRatio;
  }

  get aspectRatio(): number {
    return this._aspectRatio;
  }

  set outputs(outputs: string[]) {
    this._outputs = outputs;
  }

  get outputs(): string[] {
    return this._outputs ? this._outputs : [];
  }

  get reLayoutSignalInstance(): Signal<ContentHandler, void> {
    return this._reLayout;
  }

  get privateCopy(): WidgetModel {
    return this._privateCopy;
  }

  get nodeAddedSignal(): Signal<ContentHandler, INode> {
    return this._nodeAdded;
  }

  get contentChanged(): Signal<ContentHandler, IChartInput> {
    return this._contentChanged;
  }

  get contentReset(): Signal<ContentHandler, IChartInput> {
    return this._contentReset;
  }

  get includeContent(): Signal<ContentHandler, IChartInput> {
    return this._includeContent;
  }

  get chartStateUpdate(): Signal<ContentHandler, IChartInput> {
    return this._chartStateUpdate;
  }

  get sizeStateUpdate(): Signal<ContentHandler, IChartInput> {
    return this._sizeStateUpdate;
  }

  setPrivateCopy(widgetModel: WidgetModel): void {
    if (!widgetModel) {
      return;
    }
    this._privateCopy = widgetModel;
  }

  constructor(context: DocumentRegistry.Context) {
    this.context = context;
    //this.context.model.contentChanged.connect(this._onContentChanged, this);
    //this.emit();
  }

  /**
   * Write graph info into the model
   * @param content
   */
  public update(content: string): void {
    if (this.context) {
      this.context.model.fromString(content);
    }
  }

  /**
   *
   * Load the file from the model
   * And covert it to include UI elements
   * Send to render
   * @param width, the width of the svg area
   * @param height, the width of the svg area
   */
  public emit(width: number, height: number): void {
    this._onContentChanged(width, height);
  }

  /**
   * Use the raw object from the file as input,  add UI information (UI schema, required, output_meta etc) to it
   *
   * @param objContent
   * @param width
   * @param height
   */
  public async renderGraph(
    objContent: any,
    width?: number,
    height?: number
  ): Promise<void> {
    const jsonString = JSON.stringify(objContent);
    // this.context.model.contentChanged.connect(this._onContentChanged, this);
    const workflows: IChartInput = await requestAPI<any>('load_graph', {
      body: jsonString,
      method: 'POST'
    });
    if (width) {
      workflows.width = width;
    }
    if (height) {
      workflows.height = height;
    }
    this._contentChanged.emit(workflows);
  }

  private _onContentChanged(width: number, height: number): void {
    console.log('content chagned');
    const refreshContent = async (): Promise<void> => {
      if (this.context === null) {
        return;
      }
      await this.context.ready;
      const yamlContent = this.context.model.toString();
      console.log('model path', this.context.path);
      const objContent = jsyaml.safeLoad(yamlContent);
      this.renderGraph(objContent, width, height);
    };
    refreshContent();
  }
}

export interface IPort {
  name: string;
  type: string[][];
  dynamic?: boolean;
}

export interface INode {
  parentIds?: string[];
  width?: number;
  x?: number;
  y?: number;
  id: string;
  type: string;
  schema: any;
  ui: any;
  conf: any;
  inputs: IPort[];
  outputs: IPort[];
  filepath?: string;
  module?: string;
  required: any;
  output_meta: any;
  busy?: boolean;
}

export interface IEdge {
  from: string;
  to: string;
}

export interface IChartInput {
  nodes: INode[];
  edges: IEdge[];
  width?: number;
  height?: number;
}

export interface IAllNodes {
  [key: string]: INode[];
}

export class GreenflowWidget extends DocumentWidget<MainAreaWidget<MainView>> {
  constructor(options: DocumentWidget.IOptions<MainAreaWidget<MainView>>) {
    super({ ...options });
    this.context = options['context'];
  }

  get contentHandler(): ContentHandler {
    const mainAreaWidget = this.content;
    const mainView = mainAreaWidget.content;
    return mainView.contentHandler;
  }

  readonly context: DocumentRegistry.Context;
}

/**
 * A widget factory for drawio.
 */
export class GreenflowFactory extends ABCWidgetFactory<
  GreenflowWidget,
  DocumentRegistry.IModel
> {
  /**
   * Create a new widget given a context.
   */
  constructor(options: DocumentRegistry.IWidgetFactoryOptions) {
    super(options);
  }

  protected createNewWidget(context: DocumentRegistry.Context): GreenflowWidget {
    const contentHandler = new ContentHandler(context);
    const mainView = new MainView(contentHandler);
    const widget = new MainAreaWidget<MainView>({ content: mainView });
    widget.toolbar.hide();
    return new GreenflowWidget({ context, content: widget });
  }
}

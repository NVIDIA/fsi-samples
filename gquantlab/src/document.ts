import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './gquantlab';
import YAML from 'yaml';
import { Signal, ISignal } from '@lumino/signaling';
import { MainAreaWidget } from '@jupyterlab/apputils';
import { WidgetModel } from '@jupyter-widgets/base';

export class ContentHandler {
  context: DocumentRegistry.Context;
  // this one will relayout the graph
  private _contentChanged = new Signal<ContentHandler, IChartInput>(this);
  // this one just update the graphs, no relayout
  private _chartStateUpdate = new Signal<ContentHandler, IChartInput>(this);

  // create a signal that emits the added node command
  private _nodeAdded = new Signal<ContentHandler, INode>(this);

  private _privateCopy: WidgetModel;

  // create a signal that emits the relayout command
  private _reLayout = new Signal<ContentHandler, void>(this);

  private _aspectRatio: number = 0.5;

  get reLayoutSignal(): Signal<ContentHandler, void> {
    return this._reLayout;
  }

  set aspectRatio(aspectRatio: number) {
    this._aspectRatio = aspectRatio;
  }

  get aspectRatio() {
    return this._aspectRatio;
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

  get contentChanged(): ISignal<ContentHandler, IChartInput> {
    return this._contentChanged;
  }

  get chartStateUpdate(): Signal<ContentHandler, IChartInput> {
    return this._chartStateUpdate;
  }


  setPrivateCopy(widgetModel: WidgetModel): void {
    if (!widgetModel) {
      return;
    }
    this._privateCopy = widgetModel;
  }

  renderNodesAndEdges(workflows: IChartInput): void {
    this._contentChanged.emit(workflows);
  }

  constructor(context: DocumentRegistry.Context) {
    this.context = context;
    //this.context.model.contentChanged.connect(this._onContentChanged, this);
    this.emit();
  }

  public update(content: string): void {
    if (this.context) {
      this.context.model.fromString(content);
    }
  }

  public emit(): void {
    this._onContentChanged();
  }

  public async renderGraph(objContent: any, width?: number, height?: number): Promise<void> {
      const jsonString = JSON.stringify(objContent);
    // this.context.model.contentChanged.connect(this._onContentChanged, this);
    const workflows: IChartInput = await requestAPI<any>('load_graph', {
      body: jsonString,
      method: 'POST'
    });
    if (width){
      workflows.width  = width;
    }
    if (height){
      workflows.height = height;
    }
    this.renderNodesAndEdges(workflows);
  }

  private _onContentChanged(): void {
    console.log('content chagned');
    const refreshContent = async (): Promise<void> => {
      if (this.context === null) {
        return;
      }
      await this.context.ready;
      const yamlContent = this.context.model.toString();
      console.log('model path', this.context.path);
      const objContent = YAML.parse(yamlContent);
      this.renderGraph(objContent);
    };
    refreshContent();
  }
}

export interface IPort {
  name: string;
  type: string[];
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
  required: any;
  output_columns: any;
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

export class GquantWidget extends DocumentWidget<MainAreaWidget<MainView>> {
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
export class GquantFactory extends ABCWidgetFactory<
  GquantWidget,
  DocumentRegistry.IModel
> {
  /**
   * Create a new widget given a context.
   */
  constructor(options: DocumentRegistry.IWidgetFactoryOptions) {
    super(options);
  }

  protected createNewWidget(context: DocumentRegistry.Context): GquantWidget {
    const contentHandler = new ContentHandler(context);
    const mainView = new MainView(contentHandler);
    const widget = new MainAreaWidget<MainView>({ content: mainView });
    return new GquantWidget({ context, content: widget });
  }
}

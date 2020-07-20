import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './gquantlab';
import YAML from 'yaml';
import { Signal } from '@lumino/signaling';
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

  private _aspectRatio: number = 0.3;

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

  get contentChanged(): Signal<ContentHandler, IChartInput> {
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
   * Use the raw object from the file as input,  add UI information (UI schema, required, output_columns etc) to it
   * 
   * @param objContent 
   * @param width 
   * @param height 
   */
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
      const objContent = YAML.parse(yamlContent);
      this.renderGraph(objContent, width, height);
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

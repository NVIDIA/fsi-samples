import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './gquantlab';
import YAML from 'yaml';
import { Signal, ISignal } from '@lumino/signaling';

export class ContentHandler {
  context: DocumentRegistry.Context;
  private _contentChanged = new Signal<ContentHandler, IChartInput>(this);

  get contentChanged(): ISignal<ContentHandler, IChartInput> {
    return this._contentChanged;
  }

  constructor(context: DocumentRegistry.Context) {
    this.context = context;
    this.context.model.contentChanged.connect(this._onContentChanged, this);
  }

  private _onContentChanged(): void {
    console.log('content chagned');
    const refreshContent = async (): Promise<void> => {
      await this.context.ready;
      const yamlContent = this.context.model.toString();
      const objContent = YAML.parse(yamlContent);
      const jsonString = JSON.stringify(objContent);
      this.context.model.contentChanged.connect(this._onContentChanged, this);
      const allNodes = await requestAPI<any>('all_nodes');
      const workflows = await requestAPI<any>('load_graph', {
        body: jsonString,
        method: 'POST'
      });
      this._contentChanged.emit({
        allNodes: allNodes,
        nodes: workflows['nodes'],
        edges: workflows['edges']
      });
    };
    refreshContent();
  }
}

export interface IChartInput {
  allNodes: [];
  nodes: [];
  edges: [];
}

export class GquantWidget extends DocumentWidget<MainView> {
  constructor(options: DocumentWidget.IOptions<MainView>) {
    super({ ...options });
    this.context = options['context'];
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
    return new GquantWidget({ context, content: new MainView(contentHandler) });
  }
}
